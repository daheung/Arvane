import os
import time
import inspect
import functools
import types

from io import TextIOWrapper, BufferedIOBase
from typing import Any, Dict, List, DefaultDict, Optional, Callable, Tuple
from collections import defaultdict

class PredictorAnalyzer:
    def __init__(
        self,
        target_cls: Any,
        target_instance: Optional[Any] = None,
        logger_path: str = None,
        enable_private_method: bool = False,
    ):
        self.target_cls = target_cls
        self.target_instance = target_instance
        self.logger_path = logger_path
        self.enable_private_method = enable_private_method

        self.delegate = DelegateInstExecuter(
            target_cls=target_cls,
            logger_path=logger_path,
            enable_private_method=enable_private_method,
        )

    def start_analysis(self) -> None:
        if self.target_instance:
            self.delegate.hook_instance(self.target_instance)
        else:
            self.delegate.hook_all()

    def stop_analysis(self) -> None:
        if self.target_instance:
            self.delegate.unhook_instance(self.target_instance)
        else:
            self.delegate.unhook()

class DelegateInstExecuter:
    def __init__(
        self,
        target_cls: type[Any],
        logger_path: str = None,
        enable_private_method: bool = False,
    ):
        self.target_cls = target_cls
        self.enable_private_method = enable_private_method
        self.sink: Optional[Callable[[str, float, Dict[str, Any]], str]] = None

        self._fp: Optional[TextIOWrapper[BufferedIOBase]] = (
            open(os.path.abspath(logger_path), "w") if logger_path else None
        )

        # 클래스 전체 훅용 원본 메서드 저장
        self._originals: Dict[str, Any] = {}

        # 인스턴스별 훅용 원본 메서드 메타데이터
        #  id(instance) -> { name : (had_attr, orig_value) }
        self._instance_originals: Dict[int, Dict[str, Tuple[bool, Any]]] = {}

        # 타이밍 저장
        self._timings: DefaultDict[str, List[float]] = defaultdict(list)

    def __del__(self):
        if self._fp and not self._fp.closed:
            self._fp.close()

    def close(self):
        if self._fp and not self._fp.closed:
            self._fp.close()

    # 공통 유틸
    def set_sink(
        self,
        sink: Optional[Callable[[str, float, Dict[str, Any]], str]],
    ) -> None:
        self.sink = sink

    def _emit(self, name: str, dt: float, ctx: Dict[str, Any]) -> None:
        if self.sink:
            msg = self.sink(name, dt, ctx)
            if msg is not None and self._fp:
                self._fp.write(msg)
                self._fp.flush()

    # 클래스 전체 후킹 (기존 기능: hook_all / hook / unhook)
    def hook_all(self) -> None:
        names: List[str] = []
        for n, attr in self.target_cls.__dict__.items():
            if not self.enable_private_method and n.startswith("_"):
                continue
            if inspect.isfunction(attr) or isinstance(attr, (staticmethod, classmethod)):
                names.append(n)
        self.hook(names)

    def hook(self, names: List[str]) -> None:
        for name in names:
            if name in self._originals:
                continue
            attr = self.target_cls.__dict__.get(name, getattr(self.target_cls, name, None))
            if attr is None:
                continue
            wrapped = self._make_wrapped(name, attr)
            if wrapped is not None:
                self._originals[name] = attr
                setattr(self.target_cls, name, wrapped)

    def unhook(self, names: Optional[List[str]] = None) -> None:
        names = names or list(self._originals.keys())
        for name in names:
            orig = self._originals.pop(name, None)
            if orig is not None:
                setattr(self.target_cls, name, orig)

    def _make_wrapped(self, name: str, attr: Any):
        # 인스턴스 메서드
        if inspect.isfunction(attr):
            func = attr

            @functools.wraps(func)
            def wrapped(instance, *a, **k):
                t0 = time.perf_counter()
                ok = True
                try:
                    ret = func(instance, *a, **k)
                    return ret
                except Exception:
                    ok = False
                    raise
                finally:
                    dt = time.perf_counter() - t0
                    self._timings[name].append(dt)
                    self._emit(
                        name,
                        dt,
                        {"ok": ok, "args_len": len(a), "kwargs_len": len(k)},
                    )

            return wrapped

        # 정적 메서드
        if isinstance(attr, staticmethod):
            func = attr.__func__

            @functools.wraps(func)
            def wrapped(*a, **k):
                t0 = time.perf_counter()
                ok = True
                try:
                    ret = func(*a, **k)
                    return ret
                except Exception:
                    ok = False
                    raise
                finally:
                    dt = time.perf_counter() - t0
                    self._timings[name].append(dt)
                    self._emit(
                        name,
                        dt,
                        {"ok": ok, "args_len": len(a), "kwargs_len": len(k)},
                    )

            return staticmethod(wrapped)

        # 클래스 메서드
        if isinstance(attr, classmethod):
            func = attr.__func__

            @functools.wraps(func)
            def wrapped(cls, *a, **k):
                t0 = time.perf_counter()
                ok = True
                try:
                    ret = func(cls, *a, **k)
                    return ret
                except Exception:
                    ok = False
                    raise
                finally:
                    dt = time.perf_counter() - t0
                    self._timings[name].append(dt)
                    self._emit(
                        name,
                        dt,
                        {"ok": ok, "args_len": len(a), "kwargs_len": len(k)},
                    )

            return classmethod(wrapped)

        return None

    # 특정 인스턴스만 후킹
    def _make_instance_wrapped(self, name: str, func: Callable) -> Callable:
        """
        인스턴스용 래퍼.
        - func: 클래스 __dict__ 에서 가져온 원본 함수 (unbound function)
        """

        @functools.wraps(func)
        def wrapped(inst, *a, **k):
            t0 = time.perf_counter()
            ok = True
            try:
                # unbound function 이므로 inst 를 첫 인자로 넘겨 줌
                ret = func(inst, *a, **k)
                return ret
            except Exception:
                ok = False
                raise
            finally:
                dt = time.perf_counter() - t0
                self._timings[name].append(dt)
                self._emit(
                    name,
                    dt,
                    {"ok": ok, "args_len": len(a), "kwargs_len": len(k)},
                )

        return wrapped

    def hook_instance(self, instance: object, names: Optional[List[str]] = None) -> None:
        if not isinstance(instance, self.target_cls):
            raise TypeError(
                f"hook_instance: expected instance of {self.target_cls.__name__}, "
                f"got {type(instance).__name__}"
            )

        cls = self.target_cls

        # 후킹할 메서드 이름 리스트 자동 수집
        if names is None:
            names = []
            for n, attr in cls.__dict__.items():
                if not self.enable_private_method and n.startswith("_"):
                    continue
                if inspect.isfunction(attr):
                    names.append(n)

        inst_id = id(instance)
        inst_map = self._instance_originals.setdefault(inst_id, {})

        for name in names:
            if name in inst_map:
                # 이미 이 인스턴스에서 후킹된 메서드
                continue

            attr = cls.__dict__.get(name)
            if not inspect.isfunction(attr):
                # 인스턴스 메서드가 아니면 스킵 (static/classmethod 제외)
                continue

            # 인스턴스에 원래부터 같은 이름의 attribute가 있었는지 여부
            had_attr = name in instance.__dict__
            orig_value = getattr(instance, name) if had_attr else None

            # 나중에 unhook_instance에서 복구할 수 있도록 저장
            inst_map[name] = (had_attr, orig_value)

            # 래퍼 생성 후 인스턴스에만 바인딩
            wrapped_func = self._make_instance_wrapped(name, attr)
            bound_wrapped = types.MethodType(wrapped_func, instance)

            setattr(instance, name, bound_wrapped)

    def unhook_instance(self, instance: object, names: Optional[List[str]] = None) -> None:
        inst_id = id(instance)
        inst_map = self._instance_originals.get(inst_id)
        if not inst_map:
            return

        if names is None:
            names = list(inst_map.keys())

        for name in names:
            meta = inst_map.pop(name, None)
            if meta is None:
                continue

            had_attr, orig_value = meta

            if had_attr:
                setattr(instance, name, orig_value)
            else:
                if hasattr(instance, name):
                    delattr(instance, name)

        if not inst_map:
            self._instance_originals.pop(inst_id, None)
