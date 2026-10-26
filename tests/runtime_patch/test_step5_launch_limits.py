"""The requested DP16 decoder must retain its token budget and context limit."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    bash = shutil.which("bash")
    if not bash or subprocess.run([bash, "-c", "true"], capture_output=True).returncode:
        pytest.skip("A working Bash is required for launcher checks")
    env = {k: v for k, v in os.environ.items() if not k.startswith("STEP5_")}
    env.update(STEP5_HEAD_IP="172.20.2.22", VLLM_HOST_IP="172.20.2.22",
               STEP5_TOKENIZER="/external/tokenizer", STEP5_STATE_DIR=str(tmp_path),
               STEP5_DRY_RUN="1", STEP5_EP_TABLE="/external/ep_table.json",
               HCU_STEP5_TUNING_CONFIG="/external/tuning.json")

    def run(phase, rank=0, **settings):
        return subprocess.run([bash, str(ROOT / "benchmarks/step5/launch_multinode.sh"),
                               phase, str(rank)], env=dict(env, **settings),
                              capture_output=True, text=True, timeout=15)
    return run


@pytest.mark.parametrize("rank", range(4))
def test_dp16_decode_retains_fixed_limits_and_local_ranks(launcher, rank):
    result = launcher("decode", rank, STEP5_DECODE_PARALLEL="dp16tp1",
                      STEP5_DECODE_PROFILE="dp16_native")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    expected = {"--tensor-parallel-size": "1", "--data-parallel-size": "16",
                "--data-parallel-size-local": "4", "--max-num-batched-tokens": "32768",
                "--max-model-len": "1048576", "--long-prefill-token-threshold": "2048",
                "--gpu-memory-utilization": "0.91"}
    for key, value in expected.items():
        assert command[command.index(key) + 1] == value
    assert "VLLM_HCU_DP_PACKED_AG=0" in result.stdout
    assert "HCU_STEP5_BOUNDED_EP_WORKSPACE=1" in result.stdout
    assert "VLLM_STEP4_DSA_GRAPH_REGIONS=0" in result.stdout
    if rank:
        assert command[command.index("--data-parallel-start-rank") + 1] == str(rank * 4)
        assert "--headless" in command
    else:
        assert command[command.index("--api-server-count") + 1] == "1"


def test_existing_decode_and_prefill_defaults_stay_separate(launcher):
    for phase, tp, dp, budget, context in [
        ("decode", "4", "4", "32768", "1048576"),
        ("prefill", "1", "16", "8192", "70000"),
    ]:
        settings = {} if phase == "decode" else dict(
            STEP5_DECODE_PARALLEL="dp16tp1", STEP5_DECODE_PROFILE="dp16_native",
            STEP5_DECODE_GPU_MEMORY_UTILIZATION="0.91",
            STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD="2048")
        result = launcher(phase, **settings)
        assert result.returncode == 0, result.stderr
        command = shlex.split(result.stdout.splitlines()[-1])
        for key, value in {"--tensor-parallel-size": tp, "--data-parallel-size": dp,
                           "--max-num-batched-tokens": budget, "--max-model-len": context}.items():
            assert command[command.index(key) + 1] == value
        assert "--long-prefill-token-threshold" not in command


def test_prefill_can_screen_native_deepgemm_without_changing_recipe(launcher):
    baseline = launcher("prefill")
    candidate = launcher("prefill", STEP5_PREFILL_EP_FUSED_MOE="0")
    assert baseline.returncode == candidate.returncode == 0
    assert "VLLM_HCU_EP_FUSED_MOE=1" in baseline.stdout
    assert "VLLM_HCU_EP_FUSED_MOE=0" in candidate.stdout
    baseline_command = shlex.split(baseline.stdout.splitlines()[-1])
    candidate_command = shlex.split(candidate.stdout.splitlines()[-1])
    assert baseline_command == candidate_command


def test_prefill_can_screen_shared_expert_late_stream_without_changing_recipe(launcher):
    baseline = launcher("prefill")
    candidate = launcher("prefill", STEP5_PREFILL_SHARED_STREAM_MIN_TOKENS="0")
    assert baseline.returncode == candidate.returncode == 0
    assert "HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096" in baseline.stdout
    assert "HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=0" in candidate.stdout
    baseline_command = shlex.split(baseline.stdout.splitlines()[-1])
    candidate_command = shlex.split(candidate.stdout.splitlines()[-1])
    assert baseline_command == candidate_command


def test_invalid_prefill_expert_kernel_switch_fails(launcher):
    assert launcher("prefill", STEP5_PREFILL_EP_FUSED_MOE="deepgemm").returncode != 0


def test_invalid_prefill_shared_expert_stream_threshold_fails(launcher):
    assert launcher("prefill", STEP5_PREFILL_SHARED_STREAM_MIN_TOKENS="1").returncode != 0


def test_prefill_can_select_aiter_without_changing_fixed_limits(launcher):
    result = launcher("prefill", STEP5_PREFILL_MOE_BACKEND="aiter")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--moe-backend") + 1] == "aiter"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--data-parallel-size") + 1] == "16"
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
    assert command[command.index("--max-model-len") + 1] == "70000"


def test_invalid_prefill_moe_backend_fails(launcher):
    assert launcher("prefill", STEP5_PREFILL_MOE_BACKEND="deep_gemm").returncode != 0


def test_e22_cached_asm_profile_keeps_measured_limits(launcher):
    result = launcher("prefill", STEP5_PREFILL_PROFILE="asm_g1_1021",
                      STEP5_PREFILL_PLACEMENT="/external/prefill.pt")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert "HCU_STEP5_PREFILL_ASM_G1=1 GPU_MAX_HW_QUEUES=4" in result.stdout
    assert "HCU_STEP5_DENSE_TUNING=1 HCU_STEP5_PREFILL_G2=1" in result.stdout
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
    assert command[command.index("--max-model-len") + 1] == "70000"
    assert "--enable-eplb" not in command


def test_e23_cached_asm_requires_an_explicit_fixed_map(launcher, tmp_path):
    settings = dict(STEP5_PREFILL_PROFILE="e23_asm_1021",
                    STEP5_PREFILL_PLACEMENT="/external/prefill.pt")
    assert launcher("prefill", **settings).returncode != 0
    # Dry-run checks the CLI contract only; the normal static loader validates
    # map shape, namespace and assignments before loading any model weights.
    plan = tmp_path / "static.json"
    plan.write_text("{}")
    result = launcher("prefill", **settings, STEP5_PREFILL_REDUNDANT_EXPERTS="16",
                      STEP5_PREFILL_STATIC_MAP=str(plan))
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert "--enable-eplb" in command
    assert "HCU_STEP5_PREFILL_ASM_G1=1 GPU_MAX_HW_QUEUES=8" in result.stdout
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
    assert command[command.index("--max-model-len") + 1] == "70000"


def test_decode_clears_prefill_cached_asm_switch(launcher):
    result = launcher("decode", HCU_STEP5_PREFILL_ASM_G1="1")
    assert result.returncode == 0, result.stderr
    assert "HCU_STEP5_PREFILL_ASM_G1=0 GPU_MAX_HW_QUEUES=4" in result.stdout


@pytest.mark.parametrize("settings", [
    {"STEP5_DECODE_PARALLEL": "dp8tp2"},
    {"STEP5_DECODE_PARALLEL": "dp16tp1", "STEP5_DECODE_PROFILE": "itl6479", "STEP5_DECODE_PLACEMENT": "/external/placement.pt"},
    {"STEP5_DECODE_PARALLEL": "dp16tp1", "STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD": "-1"},
])
def test_invalid_or_incompatible_decode_modes_fail_before_launch(launcher, settings):
    assert launcher("decode", **settings).returncode != 0


@pytest.mark.parametrize("profile,g1,g2", [
    ("dense_1021", "0", "0"), ("g2_1021", "0", "1"),
])
def test_measured_prefill_profiles_preserve_limits_and_effective_knobs(launcher, profile, g1, g2):
    result = launcher("prefill", STEP5_PREFILL_PROFILE=profile,
                      STEP5_PREFILL_PLACEMENT="/external/prefill.pt")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    for key, value in {"--tensor-parallel-size": "1", "--data-parallel-size": "16",
                       "--max-num-batched-tokens": "8192", "--max-model-len": "70000",
                       "--long-prefill-token-threshold": "0"}.items():
        assert command[command.index(key) + 1] == value
    assert "HCU_DP_MOE_PIPE_MIN_TOKENS=65536" in result.stdout
    assert "HCU_DBO_MIN_TOKENS=4096 HCU_EP_DIRECT_SUM=65536 HCU_DP_MOE_PIPE_UNIFORM=1" in result.stdout
    assert f"HCU_STEP5_DENSE_TUNING=1 HCU_STEP5_PREFILL_G2={g2}" in result.stdout
    assert "EP_FUSED_MOE_CHUNK=65536" in result.stdout
    assert "HCU_STEP5_BOUNDED_EP_WORKSPACE=1" in result.stdout


@pytest.mark.parametrize("extra", [
    {"STEP5_PREFILL_DP_SIZE": "8"}, {"STEP5_PREFILL_BATCHED_TOKENS": "32768"},
    {"STEP5_PREFILL_MAX_MODEL_LEN": "1048576"}, {"STEP5_PREFILL_EP_FUSED_MOE": "0"},
])
def test_measured_prefill_profile_rejects_unmeasured_combinations(launcher, extra):
    result = launcher("prefill", STEP5_PREFILL_PROFILE="dense_1021",
                      STEP5_PREFILL_PLACEMENT="/external/prefill.pt", **extra)
    assert result.returncode != 0


def test_decode_fabric_profile_retains_one_million_context(launcher):
    result = launcher("decode", STEP5_DECODE_PROFILE="itl6479",
                      STEP5_DECODE_PLACEMENT="/external/decode.pt",
                      HCU_STEP5_DP_FABRIC="1", HCU_STEP5_CONTEXT_GRAPH_REGIONS="8736")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--max-model-len") + 1] == "1048576"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    assert "VLLM_HCU_DP_PACKED_AG=1" in result.stdout
    assert "VLLM_STEP4_DSA_GRAPH_REGIONS=8736" in result.stdout
    assert "HCU_STEP5_DP_FABRIC=1 HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736" in result.stdout
    assert "VLLM_HCU_DP_CUSTOM_AG=0" in result.stdout


def test_decode_fabric_rejects_other_topologies(launcher):
    result = launcher("decode", STEP5_DECODE_PARALLEL="dp16tp1",
                      STEP5_DECODE_PROFILE="dp16_native", HCU_STEP5_DP_FABRIC="1")
    assert result.returncode != 0


def test_prefill_does_not_inherit_decode_only_context_guard(launcher):
    result = launcher("prefill", HCU_STEP5_CONTEXT_GRAPH_REGIONS="8736", HCU_STEP5_DP_FABRIC="1")
    assert result.returncode == 0, result.stderr
    assert "HCU_STEP5_DP_FABRIC=0 HCU_STEP5_CONTEXT_GRAPH_REGIONS=0" in result.stdout


def test_decode_does_not_inherit_prefill_candidates(launcher):
    result = launcher("decode", HCU_STEP5_DENSE_TUNING="1", HCU_STEP5_PREFILL_G1="1",
                      HCU_STEP5_PREFILL_G2="1", HCU_DP_MOE_PIPE_UNIFORM="1")
    assert result.returncode == 0, result.stderr
    assert "HCU_STEP5_DENSE_TUNING=0 HCU_STEP5_PREFILL_G2=0" in result.stdout
    assert "HCU_DP_MOE_PIPE_UNIFORM=0" in result.stdout
