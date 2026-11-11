#!/usr/bin/env bash
# CPU entry for the V4.1 FP8 *reference* indexer source integration.
#
# Why this wrapper exists: the replay's FP4 cases are hardware-gated. The FP4
# indexer branch requires the published triton E2M1 lowering, and that lowering
# only activates on a real HCU gfx938 (`hcu_vllm_das/fp4_packer.py`,
# `activate_extension`). On a CPU host the FP4 branch therefore refuses before
# any FP8 case can run, which is why the replay used to stop at 0/6 here.
#
# This wrapper selects the documented stock rollback for the FP4 backend
# (`HCU_VLLM_FP4_INDEXER_IMPL=stock`) and delegates to the unchanged replay.
# The FP8 reference path — cache decoding, scoring, candidate-block selection,
# graph split registration and the OOT class — is what this run exercises and is
# unaffected by that switch. The FP4 branch itself stays covered by
# tests/vllm033/test_fp4_full_admission.py and test_fp4_full_chain.py, and its
# device path remains unverified on CPU.
set -euo pipefail
here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
: "${1:?Provide an output directory}"
export HCU_VLLM_FP4_INDEXER_IMPL=stock
exec bash "${here}/run_v41_fp8_indexer_cpu.sh" "$1"
