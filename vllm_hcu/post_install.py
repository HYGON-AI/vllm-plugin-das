"""Post-install helpers for applying vLLM source patches."""


def apply_post_install_patches() -> int:
    try:
        from vllm_hcu.patch_utils import _register_patches
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
