import torch

def pad_last(x: torch.Tensor, target_len: int, dim: int = 0) -> torch.Tensor:
    """
    x: tensor with shape (..., L, ...)
    dim: pad 대상 축
    """

    dim = dim % x.ndim
    L = x.size(dim)
    if L == 0:
        raise ValueError("해당 축 길이가 0이면 마지막 값을 복제할 수 없습니다.")
    if target_len <= L:
        # 슬라이스로 잘라서 반환
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(0, target_len)
        return x[tuple(sl)]

    # 인덱스 만들기: [0,1,2,...,L-1,L-1,L-1,...] 형태
    idx = torch.arange(target_len, device=x.device).clamp_max(L - 1)

    # dim 축으로 선택
    return x.index_select(dim, idx)