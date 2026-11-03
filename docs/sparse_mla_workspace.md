# Official sparse MLA prefill workspace backport

Source: vllm-project/vllm#61023, merge d6aca837c3c6b773a557448d526db219a5a5718b.
This is a vLLM 0.25.1 **core source patch**, shipped alongside the HCU plugin.
It adds the real Pydantic AttentionConfig field and reads it directly from
vllm_config.attention_config in both sparse MLA and indexer sizing.
There is no CLI interception, HCU sidecar field, or additional-config alias.

Apply once to the vLLM source/package used by your deployment, before starting
any Python/vLLM process (use a dedicated environment when sharing packages):

```bash
python tools/apply_sparse_mla_workspace_backport.py --vllm-root /usr/local/lib/python3.10/dist-packages --check
python tools/apply_sparse_mla_workspace_backport.py --vllm-root /usr/local/lib/python3.10/dist-packages
```

Installing only the plugin wheel does not apply this core patch. Unsupported or
already patched sources fail the check before any write. Apply it to every
node's environment, then restart services. To undo, use git apply -R with the
same patch from the same vllm-root directory.

Use the official interface:

```bash
--attention-config '{"sparse_mla_prefill_workspace_factor":10}'
```

The field accepts integer factors >=2. Omitted: 5 at max_model_len <=262144,
otherwise 2. Resolution occurs once in VllmConfig initialization and survives
later KV auto-fit reductions. Factor 5 restores the old row budgets.
Sparse MLA reserves factor*max_model_len BF16 rows; indexer capacity is
8*factor*max_model_len rows. Allocation and prefill chunking read the same field,
including the HCU adapter's existing PCP clone of the core builder.
At 89088 max length the default stays 5. A larger factor reserves more workspace;
this change has no measured serving-performance or model-accuracy claim.
DeepSeek-V4.1 independent FlashMLA workspace builders remain out of scope.

The optional plugin-side [workspace deduplication](sparse_mla_workspace_dedup.md)
reduces repeated KV staging within this capacity without changing the allocation.
