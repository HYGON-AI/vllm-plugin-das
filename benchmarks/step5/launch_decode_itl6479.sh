#!/usr/bin/env bash
# Native C64/MTP3 profile; the effective EP16 table is checked before startup.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rank="${1:-}"
if [[ -z "$rank" ]]; then
    case "$(hostname -s)" in
        p5c1) rank=0 ;; p5c2) rank=1 ;; p5c3) rank=2 ;; p5c4) rank=3 ;;
        *) printf 'Usage: bash %s NODE_RANK (0..3)\n' "$0" >&2; exit 2 ;;
    esac
fi
case "$rank" in 0|1|2|3) ;; *) printf 'NODE_RANK must be 0..3\n' >&2; exit 2 ;; esac
export STEP5_HEAD_IP="${STEP5_HEAD_IP:-172.20.2.22}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-172.20.2.$((22 + rank))}"
export STEP5_TOKENIZER="${STEP5_TOKENIZER:-/opt/xuwq_step5_new_20261018/tokenizer-bytelevel}"
export STEP5_STATE_DIR="${STEP5_STATE_DIR:-/var/lib/xuwq_step5_itl6479_native}"
export STEP5_CACHE_DIR="${STEP5_CACHE_DIR:-/var/cache/xuwq_step5}"
export STEP5_DECODE_PROFILE=itl6479
: "${STEP5_DECODE_PLACEMENT:?Set STEP5_DECODE_PLACEMENT to the external Decode placement .pt file}"
export STEP5_DECODE_PLACEMENT
exec bash "$repo_root/benchmarks/step5/launch_multinode.sh" decode "$rank"
