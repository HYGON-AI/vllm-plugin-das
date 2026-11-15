# SPDX-License-Identifier: Apache-2.0
"""Portable end-to-end checks; all repository fixtures are temporary."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("upgrade_preflight.py")


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.STDOUT
    ).decode("utf-8").strip()


def write(root, name, content):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def commit(root):
    git(root, "add", ".")
    return commit_index(root)


def commit_index(root):
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hcu-preflight-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.up = self.root / "official"
        self.fork = self.root / "fork"
        self.plugin = self.root / "plugin"
        self.up.mkdir()
        git(self.up, "init", "-q")
        write(self.up, "vllm/fast.py", "def fast(x): return x + 1\n")
        write(self.up, "vllm/old.py", "def kernel(x): return x\n")
        write(self.up, "vllm/router.py", "def route(x): return x\n")
        self.base = commit(self.up)
        subprocess.run(["git", "clone", "-q", str(self.up), str(self.fork)], check=True)
        write(self.fork, "vllm/fast.py", "def fast(x): return x * 2\n")
        commit(self.fork)
        write(self.up, "vllm/fast.py", "def fast(x): return x * 2\n")
        (self.up / "vllm/old.py").unlink()
        write(self.up, "vllm/common/old.py", "def kernel(x): return x\n")
        write(self.up, "vllm/router.py", "def route(x, skip_padding=False): return x\n")
        commit(self.up)
        self.plugin.mkdir()
        git(self.plugin, "init", "-q")
        write(self.plugin, "vllm_hcu/patch/patch_router.py", '''
TARGET_MODULE = "vllm.router"
PATCH_ID = "router"
TARGETS = (f"{TARGET_MODULE}.route",)
def apply():
    require_exact_signature(fn, TARGETS[0], positional=("x",))
raise RuntimeError("Repository code must never execute")
''')
        write(self.plugin, "vllm_hcu/patch/patch_old.py", '''
TARGET_MODULE = "vllm.old"
PATCH_ID = "old"
''')
        commit(self.plugin)

    def invoke(self, output, base=None, target="HEAD"):
        target_args = ["--target-ref", target] if target is not None else []
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--plugin-root", str(self.plugin),
             "--fork-root", str(self.fork), "--upstream-root", str(self.up),
             "--upstream-base", base or self.base, *target_args, "--output", str(output)],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_reports_candidates_and_drift_without_execution_or_mutation(self):
        write(self.plugin, "untracked.txt", "preserve\n")
        roots = [self.plugin, self.fork, self.up]
        before = [(git(r, "rev-parse", "HEAD"), git(r, "status", "--porcelain")) for r in roots]
        out = self.root / "report"
        run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(report["sources"]["plugin"]["working_tree_entries"], 1)
        carried = next(x for x in report["fork_deltas"] if x["path"] == "vllm/fast.py")
        self.assertEqual(carried["upstream_review"], "identical_file_candidate")
        self.assertEqual(carried["disposition"], "pending")
        adapters = {x["patch_id"]: x for x in report["adapters"]}
        self.assertEqual(adapters["old"]["module_status"], "missing")
        self.assertIn("vllm/common/old.py", adapters["old"]["relocation_candidates"])
        self.assertEqual(adapters["router"]["signature_checks"][0]["status"], "parameter_names_changed")
        contract = adapters["router"]["signature_checks"][0]["contract_comparison"]
        self.assertEqual(contract["status"], "changed_review_required")
        self.assertIn("parameters", contract["changed_fields"])
        self.assertEqual(adapters["router"]["disposition"], "pending")
        self.assertTrue((out / "preflight.md").is_file())
        after = [(git(r, "rev-parse", "HEAD"), git(r, "status", "--porcelain")) for r in roots]
        self.assertEqual(before, after)

    def test_rejects_output_inside_sources(self):
        out = self.plugin / "report"
        run = self.invoke(out)
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(out.exists())

    def test_configured_clean_or_process_filter_is_not_executed(self):
        marker = self.root / "filter-executed"
        helper = self.root / "filter.py"
        helper.write_text(
            "from pathlib import Path\nimport sys\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "if sys.argv[1] == 'process': raise SystemExit(1)\n"
            "sys.stdout.write(sys.stdin.read())\n", encoding="utf-8")
        write(self.plugin, ".gitattributes", "vllm_hcu/patch/*.py filter=probe\n")
        commit(self.plugin)
        path = self.plugin / "vllm_hcu/patch/patch_router.py"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        for kind in ("clean", "process"):
            with self.subTest(kind=kind):
                key = "filter.probe." + kind
                git(self.plugin, "config", key, shlex.join([sys.executable, str(helper), kind]))
                try:
                    out = self.root / ("report-filter-" + kind)
                    run = self.invoke(out)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertFalse(marker.exists(), "Git executed a repository filter")
                    report = json.loads((out / "preflight.json").read_text())
                    source = report["sources"]["plugin"]
                    self.assertIsNone(source["working_tree_entries"])
                    self.assertEqual(source["working_tree_status"],
                                     "unavailable_configured_filters")
                finally:
                    git(self.plugin, "config", "--unset", key)

    def test_does_not_overwrite_report(self):
        out = self.root / "report"
        out.mkdir()
        write(out, "preflight.json", "keep me")
        run = self.invoke(out)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual((out / "preflight.json").read_text(), "keep me")

    def test_missing_base_fails_without_output(self):
        out = self.root / "report"
        run = self.invoke(out, "missing-base")
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(out.exists())

    def test_requires_explicit_target(self):
        out = self.root / "report"
        run = self.invoke(out, target=None)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--target-ref", run.stderr)
        self.assertFalse(out.exists())

    def test_target_tag_branch_and_sha_control_analysis(self):
        # Synthetic refs; an older target must be read even when HEAD is newer.
        git(self.up, "tag", "fixture-release", self.base)
        git(self.up, "branch", "fixture-next")
        head = git(self.up, "rev-parse", "HEAD")
        cases = [("fixture-release", self.base, "unchanged", "names_match_only"),
                 ("fixture-next", head, "missing", "parameter_names_changed"),
                 (self.base, self.base, "unchanged", "names_match_only")]
        for i, (ref, sha, module_status, signature_status) in enumerate(cases):
            with self.subTest(ref=ref):
                out = self.root / f"report-{i}"
                run = self.invoke(out, target=ref)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
                self.assertEqual(report["sources"]["target"]["requested_ref"], ref)
                self.assertEqual(report["sources"]["target"]["sha"], sha)
                adapters = {x["patch_id"]: x for x in report["adapters"]}
                self.assertEqual(adapters["old"]["module_status"], module_status)
                self.assertEqual(adapters["router"]["signature_checks"][0]["status"], signature_status)

    def test_partial_clone_missing_blobs_does_not_fetch(self):
        git(self.plugin, "config", "uploadpack.allowFilter", "true")
        partial = self.root / "partial-plugin"
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
             self.plugin.as_uri(), str(partial)], check=True,
        )
        self.plugin = partial
        packs = partial / ".git/objects/pack"
        before = {p.name: p.read_bytes() for p in packs.iterdir()}
        out = self.root / "report"
        run = self.invoke(out)
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(out.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in packs.iterdir()})

    def test_link_and_submodule_deltas_are_not_dropped(self):
        def blob(text):
            return subprocess.check_output(
                ["git", "-C", str(self.fork), "hash-object", "-w", "--stdin"],
                input=text.encode(),
            ).decode().strip()

        old_link, new_link = blob("native-v1"), blob("native-v2")
        old_commit = git(self.fork, "rev-parse", "HEAD")
        git(self.fork, "update-index", "--add", "--cacheinfo", "120000", old_link, "vendor-link")
        git(self.fork, "update-index", "--add", "--cacheinfo", "160000", old_commit, "vendor-submodule")
        base = commit_index(self.fork)
        git(self.fork, "update-index", "--cacheinfo", "120000", new_link, "vendor-link")
        git(self.fork, "update-index", "--cacheinfo", "160000", base, "vendor-submodule")
        git(self.fork, "update-index", "--cacheinfo", "120000", new_link, "vllm/fast.py")
        commit_index(self.fork)
        out = self.root / "report"
        run = self.invoke(out, base=base)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        rows = {r["path"]: r for r in report["fork_deltas"]}
        self.assertEqual(set(rows), {"vendor-link", "vendor-submodule", "vllm/fast.py"})
        self.assertEqual(rows["vendor-link"]["fork_mode"], "120000")
        self.assertEqual(rows["vendor-submodule"]["fork_mode"], "160000")
        self.assertEqual(rows["vllm/fast.py"]["change"], "modified")
        self.assertEqual(rows["vllm/fast.py"]["base_mode"], "100644")


if __name__ == "__main__":
    unittest.main()
