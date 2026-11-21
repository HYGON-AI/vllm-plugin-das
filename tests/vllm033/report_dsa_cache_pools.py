"""CPU planning report; explicitly not a measured GPU token capacity."""
import json


def main():
    import torch
    from test_dsa_cache_pools import setup
    from hcu_vllm_das import dsa_cache_pools as pools
    from hcu_vllm_das.dsa_cache_layout import DsaLayerSplitCacheBuilder
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    budget = 10 * 1024**3
    cfg, original, independent = setup(budget=budget)
    modes = {}
    for name, builder, cost in (
        ('off', DefaultKVCacheConfigBuilder(), 116928),
        ('previous_main_and_swa_shared_pool', DsaLayerSplitCacheBuilder(), 98560),
    ):
        plan = builder.get_kv_cache_config_from_groups(cfg, original, budget // cost)
        modes[name] = dict(capacity_tokens=get_kv_cache_capacity(cfg, plan)[0],
                           actual_backing_bytes=plan.kv_cache_tensors[0].size)
    modes['independent_pools'] = dict(capacity_tokens=get_kv_cache_capacity(cfg, independent)[0],
        actual_backing_bytes=independent.kv_cache_tensors[0].size,
        usable_pages=[n - 1 for n in independent.pool_blocks],
        bytes_per_page=independent.pool_pitches, demands_per_max_length_request=independent.pool_demands,
        transport_bytes=independent.wire_bytes)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(kind='CPU planning only; GPU verification pending',
        budget_bytes=budget, max_model_len=cfg.model_config.max_model_len,
        max_in_flight_tokens=cfg.max_in_flight_tokens, modes=modes,
        ratio_to_off=modes['independent_pools']['capacity_tokens']/modes['off']['capacity_tokens']), indent=2))


if __name__ == '__main__':
    main()
