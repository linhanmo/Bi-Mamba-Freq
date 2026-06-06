
import torch

from mambapy.mamba import Mamba, MambaConfig


def _make_mamba(d_model=32, n_layers=2, d_state=16):
    # returns eval model on cpu
    cfg = MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        pscan=True,
        use_cuda=False,
    )
    return Mamba(cfg).eval()


def _empty_caches_for_model(model, B, device, dtype):
    # caches : [cache(layer) for all layers], cache : (h, inputs)
            # h : (B, ED, N)
            # inputs : (B, ED, d_conv-1)
    caches = []
    for layer in model.layers:
        cfg = layer.mixer.config
        ED = cfg.d_inner
        N = cfg.d_state
        k = max(cfg.d_conv - 1, 0)
        inputs = torch.zeros(B, ED, k, device=device, dtype=dtype)
        caches.append((None, inputs))
    return caches

def test_chunk_step_matches_forward_two_chunks():
    torch.manual_seed(0)
    B, L, D = 2, 30, 32
    model = _make_mamba(d_model=D, n_layers=2, d_state=16)

    x = torch.randn(B, L, D) # (B, L, D)
    L1 = L // 2

    with torch.no_grad():
        gold = model(x) # (B, L, D)
        caches = _empty_caches_for_model(model, B, x.device, x.dtype)
        y1, caches = model.chunk_step(x[:, :L1, :], caches)
        y2, caches = model.chunk_step(x[:, L1:, :], caches)
        got = torch.cat([y1, y2], dim=1)
    assert torch.allclose(got, gold, atol=1e-5, rtol=1e-5)


def test_chunk_step_matches_step_single_chunk():
    torch.manual_seed(0)
    B, L, D = 2, 25, 32
    model = _make_mamba(d_model=D, n_layers=2, d_state=16)

    x = torch.randn(B, L, D) # (B, L, D)

    with torch.no_grad():
        caches_seq = _empty_caches_for_model(model, B, x.device, x.dtype)
        y_list = []
        for t in range(L):
            y_t, caches_seq = model.step(x[:, t, :], caches_seq) # (B, D)
            y_list.append(y_t.unsqueeze(1))
        logits_step = torch.cat(y_list, dim=1) # (B, L, D)

        caches_chunk = _empty_caches_for_model(model, B, x.device, x.dtype)
        logits_chunk, _ = model.chunk_step(x, caches_chunk) # (B, L, D)
    assert torch.allclose(logits_chunk, logits_step, atol=1e-5, rtol=1e-5)
