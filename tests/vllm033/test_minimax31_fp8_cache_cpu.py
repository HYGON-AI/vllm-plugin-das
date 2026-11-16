"""CPU contract tests for the task-isolated new-wheel PCP FP8 cache guard."""
import json
from types import SimpleNamespace


import torch
from hcu_vllm_das.parallel import minimax31_pcp as pcp




def layer():
    side = SimpleNamespace(dtype=torch.bfloat16,
                           kv_cache=torch.empty(0, dtype=torch.bfloat16))
    return SimpleNamespace(
        kv_cache_dtype="fp8", kv_cache_torch_dtype=torch.float8_e4m3fn,
        kv_cache=torch.zeros((2, 1, 4, 256), dtype=torch.float8_e4m3fn),
        num_kv_heads=1, head_dim=128,
        indexer=SimpleNamespace(index_cache=side),
        _k_scale=torch.tensor(0.5, dtype=torch.float32),
        _v_scale=torch.tensor(2.0, dtype=torch.float32),
    )


def test_fp8_cache_guard_and_scale_lifecycle(monkeypatch):
    monkeypatch.setattr(pcp, "_EXPERIMENTAL_FP8_PCP", True)
    slots = torch.arange(4, dtype=torch.int64)
    cases = {}
    good = layer()
    pcp.validate_fp8_sparse_cache(good, slots)
    pcp.validate_fp8_sparse_cache(good, slots)
    assert good._hcu_fp8_checked == (good.kv_cache.data_ptr(), (0.5, 2.0))
    cases["valid_fp8_main_bf16_indexer"] = "pass"

    # In-place mutation keeps data_ptr; a valid but different scale invalidates
    # any KV already written. Rebinding a cache permits a fresh scale.
    original_ptr = good._k_scale.data_ptr()
    good._k_scale.fill_(0.75)
    assert good._k_scale.data_ptr() == original_ptr
    try:
        pcp.validate_fp8_sparse_cache(good, slots)
    except ValueError as e:
        assert "scales changed" in str(e)
    else:
        raise AssertionError("in-place valid scale mutation was accepted")
    cases["in_place_scale_change"] = "rejected"
    good.kv_cache = good.kv_cache.view_as(good.kv_cache)
    try:
        pcp.validate_fp8_sparse_cache(good, slots)
    except ValueError as e:
        assert "scales changed" in str(e)
    else:
        raise AssertionError("same-storage alias bypassed scale guard")
    cases["same_storage_alias"] = "rejected"
    old_cache = good.kv_cache
    good.kv_cache = torch.zeros_like(old_cache)
    pcp.validate_fp8_sparse_cache(good, slots)
    cases["new_cache_new_scale"] = "pass"


    def rejects(name, edit):
        obj = layer()
        edit(obj)
        try:
            pcp.validate_fp8_sparse_cache(obj, slots)
        except ValueError:
            cases[name] = "rejected"
        else:
            raise AssertionError(name + " was accepted")


    rejects("wrong_physical_dtype", lambda o: setattr(o, "kv_cache", torch.zeros((2, 1, 4, 256), dtype=torch.bfloat16)))
    rejects("wrong_cache_shape", lambda o: setattr(o, "kv_cache", torch.zeros((2, 1, 4, 128), dtype=torch.float8_e4m3fn)))
    rejects("fp8_indexer_unsupported", lambda o: setattr(o.indexer.index_cache, "dtype", torch.float8_e4m3fn))
    rejects("non_scalar_scale", lambda o: setattr(o, "_k_scale", torch.ones(2)))
    rejects("zero_scale", lambda o: setattr(o, "_k_scale", torch.tensor(0.0)))
    rejects("nan_scale", lambda o: setattr(o, "_k_scale", torch.tensor(float("nan"))))
    rejects("wrong_scale_dtype", lambda o: setattr(o, "_k_scale", torch.tensor(1.0, dtype=torch.bfloat16)))
    rejects("missing_scale", lambda o: delattr(o, "_k_scale"))

    bf16 = layer()
    bf16.kv_cache_dtype = "bfloat16"
    pcp.validate_fp8_sparse_cache(bf16, slots)
    cases["bf16_path_unchanged"] = "pass"
    print(json.dumps({"plugin_file": pcp.__file__, "cases": cases}, sort_keys=True))
