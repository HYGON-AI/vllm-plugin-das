"""Post-install helpers for applying vLLM source patches."""


def apply_post_install_patches() -> int:
    try:
        from pathlib import Path

        import vllm_hcu
        from vllm_hcu.patches import _module_to_source_path
        from vllm_hcu.patch_utils import _register_patches
        #fused_moe_modular_method等在import时会缓存 FusedMoEKernel，import_hook替换modular_kernel不可靠，改用软链。
        dst = _module_to_source_path(
            "vllm.model_executor.layers.fused_moe.modular_kernel"
        )
        src = (
            Path(vllm_hcu.__file__).parent
            / "model_executor/layers/fused_moe/modular_kernel.py"
        ).resolve()
        if dst is None:
            raise FileNotFoundError("vllm modular_kernel not found")
        link_path = dst
        if not (link_path.is_symlink() and link_path.resolve() == src):
            link_path.unlink(missing_ok=True)
            link_path.symlink_to(src)

        _register_patches()
        print("Post-install patches applied successfully")
        return 0
    except Exception as e:
        print(f"Warning: Failed to apply patches: {e}")
        return 1


def main() -> int:
    return apply_post_install_patches()


if __name__ == "__main__":
    raise SystemExit(main())
