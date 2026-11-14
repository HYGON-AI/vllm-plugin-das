# Multi-node distributed — reserved

This layer cannot be executed in the current single-machine environment.
Future tests must use `multi_node` and `node_count(...)` markers and require
real cross-node resources.

Intended scope includes cross-node TP/PP/DP/EP, DeepEP, Mooncake/RDMA,
bootstrap, network failure, remote restart, and scaling.

The TP16 RCCL eager/graph regression runs on two eight-GPU nodes. On both
nodes set `MASTER_ADDR` to the first node's address, and set `NODE_RANK` to
0 or 1 respectively:

```bash
NCCL_PROTO=Simple torchrun --nnodes 2 --nproc-per-node 8 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29578 \
  tests/distributed/multi_node/test_pynccl_graph_ownership.py --capture-gather
```

This captures 12 batch sizes, each with 185 PyNccl all-reduces, then runs
200 iterations mixing eager/main-graph work, three draft graph replays,
captured draft logits all-gather, eager Torch logits all-gathers, and
asynchronous output copies. Omit --capture-gather to exercise eager-only
draft logits as in PIECEWISE mode. Inputs change
between replays; gathered values and final all-reduce outputs are checked.
For a controlled reproduction on an affected RCCL build, append
`--unpatched` to the command. That path can hang; the collective watchdog is set to 120 seconds,
but a blocked driver call can also prevent timely watchdog termination. Normal pytest collection skips this regression
unless it runs inside an existing two-node torchrun job.
