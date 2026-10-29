# Sparse MLA prefill workspace (vLLM #61023 backport)

Backported from vllm-project/vllm#61023, merge d6aca837c3c6b773a557448d526db219a5a5718b.
The v0.25.1 plugin uses its existing HCU configuration sidecar instead of changing
installed vLLM dataclass schemas:

```bash
--additional-config '{"hcu":{"sparse_mla_prefill_workspace_factor":2}}'
```

The integer factor must be >=2. Omitted: 5 at max_model_len <=262144, otherwise 2.
It resolves once at VllmConfig initialization and survives serialization and
later KV auto-fit changes to max_model_len. Explicit 5 restores old capacities.
Sparse FlashMLA reserves factor*max_model_len BF16 rows; indexer capacity is
8*factor*max_model_len rows. Allocation and both normal/PCP prefill chunking
use the same policy. Existing chunkers and attention kernels are unchanged.

At 89088 max length the default stays 5. Explicit 2 reduces these workspace
row budgets by 60%; this is not a measured increase in free KV memory or decode
throughput. Concurrent long prefills can require more gather passes.
This change does not tune DeepSeek-V4.1's independent workspace builders.
