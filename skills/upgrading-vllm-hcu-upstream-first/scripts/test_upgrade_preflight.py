# SPDX-License-Identifier: Apache-2.0
"""Portable end-to-end checks; all repository fixtures are temporary."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        # Fixture repositories must not inherit runner filters or user settings.
        self.git_config = patch.dict(os.environ, {
            "GIT_CONFIG_GLOBAL": str(self.root / "global.gitconfig"),
            "GIT_CONFIG_NOSYSTEM": "1",
        })
        self.git_config.start()
        self.addCleanup(self.git_config.stop)
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
            [sys.executable, "-I", "-S", "-B", str(SCRIPT), "--plugin-root", str(self.plugin),
             "--fork-root", str(self.fork), "--upstream-root", str(self.up),
             "--upstream-base", base or self.base, *target_args, "--output", str(output)],
            capture_output=True, text=True, encoding="utf-8",
        )

    def test_committed_python_source_encodings_preserve_body_drift(self):
        cases = [
            (b"# coding: latin-1\ndef route(x): return '\xe9'\n",
             b"# coding: latin-1\ndef route(x): return '\xf6'\n",
             "changed_review_required"),
            (b"# coding: latin-1\ndef route(x): return '\xe9'\n",
             "def route(x): return 'é'\n".encode("utf-8"),
             "unchanged_static_only"),
            (b"\xef\xbb\xbfdef route(x): return x\n",
             b"def route(x): return x\n", "unchanged_static_only"),
        ]
        for index, (old, new, expected) in enumerate(cases):
            with self.subTest(index=index):
                comment = f"# fixture {index}\n".encode("ascii")
                (self.fork / "vllm/router.py").write_bytes(old + comment)
                (self.up / "vllm/router.py").write_bytes(new + comment)
                commit(self.fork)
                commit(self.up)
                out = self.root / f"encoding-{index}"
                run = self.invoke(out)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / "preflight.json").read_text())
                row = next(r for r in report["adapters"] if r["patch_id"] == "router")
                comparison = row["signature_checks"][0]["contract_comparison"]
                self.assertEqual(comparison["status"], expected)
                self.assertEqual(comparison["changed_fields"],
                                 ["body_sha256"] if expected == "changed_review_required" else [])

    def test_rename_counts_one_worktree_entry_with_other_changes(self):
        git(self.plugin, "mv", "vllm_hcu/patch/patch_old.py",
            "vllm_hcu/patch/renamed\nold.py")
        path = self.plugin / "vllm_hcu/patch/patch_router.py"
        path.write_text(path.read_text() + "# uncommitted\n", encoding="utf-8")
        write(self.plugin, "untracked.txt", "preserve\n")
        index = (self.plugin / ".git/index").read_bytes()
        head = git(self.plugin, "rev-parse", "HEAD")
        out = self.root / "report"
        run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(report["sources"]["plugin"]["working_tree_entries"], 3)
        self.assertEqual((self.plugin / ".git/index").read_bytes(), index)
        self.assertEqual(git(self.plugin, "rev-parse", "HEAD"), head)

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

    def test_function_local_helpers_are_not_public_dotted_attributes(self):
        write(self.fork, "vllm/router.py",
              "class Ops:\n    def route(x): return x\n")
        commit(self.fork)
        write(self.plugin, "vllm_hcu/patch/patch_router.py", '''
TARGET_MODULE = "vllm.router"
PATCH_ID = "router"
def apply():
    require_exact_signature(fn, "vllm.router.Ops.route", positional=("x",))
''')
        commit(self.plugin)
        for prefix in ("def", "async def"):
            with self.subTest(prefix=prefix):
                write(self.up, "vllm/router.py",
                      f"{prefix} Ops():\n    def route(x): return x\n"
                      "    return route\n")
                commit(self.up)
                out = self.root / ("factory-" + prefix.replace(" ", "-"))
                run = self.invoke(out)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / "preflight.json").read_text())
                row = next(x for x in report["adapters"] if x["patch_id"] == "router")
                check = row["signature_checks"][0]
                self.assertEqual(check["status"], "symbol_unresolved")
                self.assertEqual(check["contract_comparison"]["status"], "symbol_unresolved")

    def test_rejects_output_inside_separate_and_shared_git_storage(self):
        original = self.plugin
        separate = self.root / "separate-worktree"
        storage = self.root / "separate-storage"
        subprocess.run(["git", "clone", "-q", "--separate-git-dir", str(storage),
                        str(original), str(separate)], check=True)
        linked = self.root / "linked-worktree"
        git(original, "worktree", "add", "--detach", str(linked), "HEAD")
        for root in (separate, linked):
            self.plugin = root
            for option in ("--git-dir", "--git-common-dir"):
                with self.subTest(worktree=root.name, storage=option):
                    git_storage = (root / git(root, "rev-parse", option)).resolve()
                    before = {str(p.relative_to(git_storage)): p.read_bytes()
                              for p in git_storage.rglob("*") if p.is_file()}
                    out = git_storage / ("refs/heads/report" + option)
                    run = self.invoke(out)
                    self.assertNotEqual(run.returncode, 0,
                                        "Preflight wrote a report into Git storage")
                    self.assertIn("Output must be outside", run.stderr)
                    self.assertFalse(out.exists())
                    after = {str(p.relative_to(git_storage)): p.read_bytes()
                             for p in git_storage.rglob("*") if p.is_file()}
                    self.assertEqual(before, after)
        self.plugin = original

    def test_package_initializer_takes_precedence_over_sibling_module(self):
        for root in (self.fork, self.up):
            write(root, "vllm/router.py", "def route(x): return x\n")
        write(self.fork, "vllm/router/__init__.py", "def route(x): return x\n")
        write(self.up, "vllm/router/__init__.py",
              "def route(x, required): return x + required\n"
              "raise RuntimeError('Repository code must never execute')\n")
        commit(self.fork)
        commit(self.up)
        out = self.root / "package-precedence"
        run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((out / "preflight.json").read_text())
        router = next(x for x in report["adapters"] if x["patch_id"] == "router")
        self.assertEqual(router["fork_path"], "vllm/router/__init__.py")
        self.assertEqual(router["official_path"], "vllm/router/__init__.py")
        check = router["signature_checks"][0]
        self.assertEqual(check["status"], "parameter_names_changed")
        self.assertEqual(check["contract_comparison"]["status"], "changed_review_required")
        self.assertIn("parameters", check["contract_comparison"]["changed_fields"])

    def test_parent_module_blocks_namespace_children_but_package_wins(self):
        for root in (self.fork, self.up):
            (root / "vllm/router.py").unlink()
            write(root, "vllm/router/ops.py", "def route(x): return x\n")
            commit(root)
        write(self.up, "vllm/router.py", "raise RuntimeError('Do not import')\n")
        commit(self.up)
        write(self.plugin, "vllm_hcu/patch/patch_router.py", '''
TARGET_MODULE = "vllm.router.ops"
PATCH_ID = "router"
def apply():
    require_exact_signature(fn, "vllm.router.ops.route", positional=("x",))
''')
        commit(self.plugin)
        for package, expected in ((False, "missing"), (True, "unchanged")):
            with self.subTest(package=package):
                if package:
                    write(self.up, "vllm/router/__init__.py",
                          "raise RuntimeError('Do not import')\n")
                    commit(self.up)
                out = self.root / ("parent-package" if package else "parent-module")
                run = self.invoke(out)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / "preflight.json").read_text())
                router = next(x for x in report["adapters"] if x["patch_id"] == "router")
                self.assertEqual(router["fork_path"], "vllm/router/ops.py")
                self.assertEqual(router["module_status"], expected)
                self.assertEqual(router["official_path"],
                                 "vllm/router/ops.py" if package else None)
                check = router["signature_checks"][0]
                self.assertEqual(check["status"],
                                 "names_match_only" if package else "symbol_unresolved")
                self.assertEqual(check["contract_comparison"]["status"],
                                 "unchanged_static_only" if package else "symbol_unresolved")

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
        for kind, scope in ((kind, scope) for kind in ("clean", "process")
                            for scope in ("--local", "--global")):
            with self.subTest(kind=kind, scope=scope):
                key = "filter.probe." + kind
                git(self.plugin, "config", scope, key,
                    shlex.join([sys.executable, str(helper), kind]))
                try:
                    out = self.root / ("report-filter-" + kind + scope)
                    run = self.invoke(out)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertFalse(marker.exists(), "Git executed a repository filter")
                    report = json.loads((out / "preflight.json").read_text())
                    source = report["sources"]["plugin"]
                    self.assertIsNone(source["working_tree_entries"])
                    self.assertEqual(source["working_tree_status"],
                                     "unavailable_configured_filters")
                finally:
                    git(self.plugin, "config", scope, "--unset", key)

    def test_classmethod_guards_preserve_bound_unbound_and_unknown_binding(self):
        source = "class Ops:\n    @classmethod\n    def ready(cls): return True\n"
        for root in (self.fork, self.up):
            write(root, "vllm/router.py", source)
            commit(root)
        for index, (expression, expected, status) in enumerate([
            ("Ops.ready", (), "names_match_only"),
            ("Ops.ready.__func__", ("cls",), "names_match_only"),
            ("fn", (), "binding_unresolved"),
            ("Ops.ready", ("missing",), "parameter_names_changed"),
        ]):
            with self.subTest(expression=expression, expected=expected):
                write(self.plugin, "vllm_hcu/patch/patch_router.py", f'''
TARGET_MODULE = "vllm.router"
PATCH_ID = "router"
def apply():
    require_exact_signature({expression}, "vllm.router.Ops.ready", positional={expected!r})
raise RuntimeError("Repository code must never execute")
''')
                commit(self.plugin)
                out = self.root / f"classmethod-{index}"
                run = self.invoke(out)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((out / "preflight.json").read_text())
                router = next(x for x in report["adapters"] if x["patch_id"] == "router")
                check = router["signature_checks"][0]
                self.assertEqual(check["status"], status)
                self.assertEqual(check["contract_comparison"]["status"], "unchanged_static_only")

    @unittest.skipUnless(os.name == "posix", "Git byte filenames require a POSIX filesystem")
    def test_non_utf8_tree_paths_are_preserved_in_valid_reports(self):
        name = os.fsdecode(b"vllm/odd-\xff.py")
        write(self.fork, name, "def route(x): return x\n")
        commit(self.fork)
        patch_name = os.fsdecode(b"vllm_hcu/patch/patch_\xfe.py")
        write(self.plugin, patch_name, 'TARGET_MODULE = "vllm.router"\n')
        commit(self.plugin)
        out = self.root / "byte-paths"
        run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
        self.assertIn(name, [row["path"] for row in report["fork_deltas"]])
        self.assertIn(patch_name, [row["adapter"] for row in report["adapters"]])
        markdown = (out / "preflight.md").read_text(encoding="utf-8")
        self.assertIn("odd-\\udcff.py", markdown)
        self.assertIn("patch_\\udcfe.py", markdown)

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

    def old_git_environment(self):
        directory = self.root / "old-git-bin"
        directory.mkdir()
        wrapper = directory / "git"
        wrapper.write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "--no-lazy-fetch" ]; then exit 129; fi\n'
            f'exec {shlex.quote(shutil.which("git"))} "$@"\n'
        )
        wrapper.chmod(0o755)
        return {"PATH": str(directory) + os.pathsep + os.environ["PATH"]}

    def test_old_git_reads_complete_local_trees(self):
        out = self.root / "report"
        with patch.dict(os.environ, self.old_git_environment()):
            run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue((out / "preflight.json").is_file())

    def test_old_git_refuses_promisor_before_fetch_or_report(self):
        git(self.plugin, "config", "remote.origin.promisor", "true")
        # Even a fully hydrated promisor checkout must fail before any object
        # access on old Git; an ignored environment switch is insufficient.
        before = git(self.plugin, "count-objects", "-v")
        out = self.root / "report"
        with patch.dict(os.environ, self.old_git_environment()):
            run = self.invoke(out)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("require a newer Git", run.stderr)
        self.assertFalse(out.exists())
        self.assertEqual(before, git(self.plugin, "count-objects", "-v"))

    def test_old_git_does_not_fetch_initialized_partial_submodule(self):
        git(self.up, "config", "uploadpack.allowFilter", "true")
        child = self.plugin / "vendor"
        subprocess.run(
            ["git", "clone", "-q", "--filter=tree:0", "--no-checkout",
             self.up.as_uri(), str(child)], check=True,
        )
        write(self.plugin, ".gitmodules",
              f'[submodule "vendor"]\n\tpath = vendor\n\turl = {self.up.as_uri()}\n')
        git(self.plugin, "add", ".gitmodules")
        git(self.plugin, "update-index", "--add", "--cacheinfo", "160000",
            git(child, "rev-parse", "HEAD"), "vendor")
        commit_index(self.plugin)
        packs = child / ".git/objects/pack"
        before = {p.name: p.read_bytes() for p in packs.iterdir()}
        env = self.old_git_environment()
        wrapper = Path(env["PATH"].split(os.pathsep)[0]) / "git"
        wrapper.write_text(wrapper.read_text().replace(
            "exec ", "unset GIT_NO_LAZY_FETCH\nexec ", 1))
        out = self.root / "report"
        with patch.dict(os.environ, env):
            run = self.invoke(out)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in packs.iterdir()},
                         "Git status fetched objects inside the submodule")
        report = json.loads((out / "preflight.json").read_text())
        self.assertEqual(report["sources"]["plugin"]["working_tree_status"], "observed")

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
