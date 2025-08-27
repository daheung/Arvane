import gc
import csv
import time
import torch
import memory
import psutil
import logging

import contextlib, time, torch
import torch._dynamo as dynamo
import torch._dynamo.utils as dutils

from typing import Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArvaneAnalyzer")
handler = logging.FileHandler("arvane_analyzer.log")
logger.addHandler(handler)

class ArvaneAnalyzer(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        track=("forward", "depth_encoder.forward"),
        log_every=10,
        profile_dir=None,
    ):
        super().__init__()
        self._delegate = model
        self._track = track
        self._log_every = log_every
        self._call_hist = defaultdict(list)
        self.t = time.time()

        self._log_fh   = open(profile_dir + "." + str(self.t), "a", newline="", buffering=1)
        self._csv_out  = csv.writer(self._log_fh)
        if self._log_fh.tell() == 0:           # 새 파일이면 헤더 쓰기
            self._csv_out.writerow([
                "timestamp", "method", "elapsed_ms",
                "current_mem_mb", "peak_mem_mb", "recompiled"
            ])

        if profile_dir is None:
            self._prof_ctx_factory = contextlib.nullcontext
        else:
            def _factory():
                return torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1,
                                                     active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                )
            self._prof_ctx_factory = _factory

        self._wrap_targets()

    def __del__(self):
        if hasattr(self, "_log_fh" + str(self.t)) and not self._log_fh.closed:
            self._log_fh.close()

    # ---------- 기본 연산 ----------
    def forward(self, *args, **kwargs):
        with self._prof_ctx_factory():
            return self._delegate(*args, **kwargs)

    @property
    def executor(self) -> torch.nn.Module:
        return self._delegate
    
    # ---------- 내부 util ----------
    def _wrap_targets(self):
        for dotted in self._track:
            owner, fn_name = self._locate(dotted)
            original = getattr(owner, fn_name)

            @torch.compiler.allow_in_graph
            def _mark() -> float:
                return time.time()
            
            @torch.compiler.allow_in_graph
            def _print(    
                *values: object,
                sep: str | None = " ",
                end: str | None = "\n",
                file: Any | None = None,
                flush: Any = False
            ) -> None:
                print(*values, sep=sep, end=end, file=file, flush=flush)
                
            def _now():
                torch.cuda.synchronize()
                return _mark()
            
            def _benchmark(orig_fn, *a, **kw):
                torch.cuda.reset_peak_memory_stats()
                before_counter = dutils.counters["stats"]["unique_graphs"]

                start = _now()
                out = orig_fn(*a, **kw)
                end = _now()

                after_counter = dutils.counters["stats"]["unique_graphs"]
                elapsed = end - start
                
                cur_mem  = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                recompile = (after_counter - before_counter) > 0

                return out, elapsed, cur_mem, peak_mem, recompile
            
            def make_wrapper(orig_fn, dotted_name=dotted):
                def _wrap(*a, **kw):
                    out, elapsed, cur_mem, peak_mem, recompile = _benchmark(orig_fn, *a, **kw)

                    h = self._call_hist[dotted_name]; h.append(elapsed)
                    if recompile:
                        logger.warning(f"[WARNING] {dotted_name} recompiled, "
                              f"unique graphs: {dutils.counters['stats']['unique_graphs']}")
                        
                    if len(h) % self._log_every == 0:
                        # _print(f"[DEBUG] {dotted_name}: {elapsed*1e3:.2f} ms "
                        #       f"(avg {sum(h)/len(h)*1e3:.2f} ms, n={len(h)})")
                        logger.info(f"[DEBUG] {dotted_name}: {elapsed*1e3:.2f} ms "
                                    f"(avg {sum(h)/len(h)*1e3:.2f} ms, n={len(h)})")
                        
                        self._csv_out.writerow([
                            f"{time.time():.6f}",
                            dotted_name,
                            f"{elapsed*1e3:.3f}",
                            f"{cur_mem/1e6:.3f}",
                            f"{peak_mem/1e6:.3f}",
                            int(recompile),
                        ])

                    return out
                return _wrap

            setattr(owner, fn_name, make_wrapper(original))

    def _locate(self, dotted):
        obj = self._delegate
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    # ---------- 통계 ----------
    def get_stats(self):
        return {k: sum(v) / len(v) for k, v in self._call_hist.items()}


# class Analyzer:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         track: list[str] = ['cpu', 'gpu', 'memory', 'gc'],
#         logs: bool = True
#     ):
#         ...

#     class ModelAnalyzer:
#         def __init__(
#             self, 
#             model, 
#             track, 
#             logs
#         ):
#             self.executor = model
#             ...
        
#         def __