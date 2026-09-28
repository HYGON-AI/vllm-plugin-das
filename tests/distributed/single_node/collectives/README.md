# Single-node collectives

Reserved for real local all-reduce, all-gather, all-to-all, timeout, and
teardown validation.

Run the AITER changing-input Graph regression on four visible HCU devices:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 pytest -q -s \
  tests/distributed/single_node/collectives/test_aiter_custom_all_reduce_graph.py
```
