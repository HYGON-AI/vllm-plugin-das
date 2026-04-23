# post_install.py
import sys
from pathlib import Path

def main():
    ROOT = Path(__file__).parent
    vllm_hcu_path = str(ROOT / "vllm_hcu")
    if vllm_hcu_path not in sys.path:
        sys.path.insert(0, vllm_hcu_path)
    
    try:
        from vllm_hcu.patch_utils import  _register_patches
        _register_patches()
        print("Post-install patches applied successfully")
    except Exception as e:
        print(f"Warning: Failed to apply patches: {e}")

if __name__ == "__main__":
    main()