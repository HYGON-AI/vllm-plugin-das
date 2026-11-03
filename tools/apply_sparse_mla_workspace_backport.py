# SPDX-License-Identifier: Apache-2.0
"""Apply the vLLM #61023 source backport before starting any vLLM process."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', type=Path, required=True,
                        help='Directory containing the vllm/ package')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.vllm_root.resolve(strict=True)
    if not (root / 'vllm/config/attention.py').is_file():
        parser.error('--vllm-root must contain vllm/config/attention.py')
    patch = Path(__file__).resolve().parents[1] / 'patches/vllm-0.25.1-sparse-mla-workspace-61023.patch'
    command = ['git', 'apply', '--check', str(patch)]
    subprocess.run(command, cwd=root, check=True)
    if not args.check:
        subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
    print('Core source backport checked' if args.check else 'Core source backport applied; restart vLLM processes')


if __name__ == '__main__':
    main()
