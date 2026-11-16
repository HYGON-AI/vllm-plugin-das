# SPDX-License-Identifier: Apache-2.0
"""Read committed Git trees without importing repository code or changing refs.

Static findings are review candidates, never automatic removal/acceptance decisions.
Requires Python 3.10+; partial clones require Git with --no-lazy-fetch support.
Older Git can inspect fully local repositories; no third-party dependencies.
"""
import argparse
import ast
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from functools import lru_cache


def git_environment():
    # Git's repository-local selectors override -C, including object/index
    # locations. Each requested root must select its own repository and storage.
    local_selectors = {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_SHALLOW_FILE", "GIT_NAMESPACE", "GIT_PREFIX",
    }
    env = {key: value for key, value in os.environ.items() if key not in local_selectors}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    return env


@lru_cache(maxsize=1)
def supports_no_lazy_fetch():
    result = subprocess.run(
        ["git", "--no-lazy-fetch", "--version"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=git_environment(), timeout=10,
    )
    return result.returncode == 0


def git(root, *args, data=None):
    env = git_environment()
    offline_args = ["--no-lazy-fetch"] if supports_no_lazy_fetch() else []
    if not offline_args:
        # Old Git ignores GIT_NO_LAZY_FETCH. Refuse promisor repositories before
        # reading objects rather than retrying an object read that could fetch.
        config = subprocess.run(
            ["git", "-C", str(root), "config", "--name-only", "--get-regexp",
             r"^(extensions\.partialclone|remote\..*\.promisor)$"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=10,
        )
        if config.returncode == 0:
            raise ValueError("This Git lacks --no-lazy-fetch; partial/promisor "
                             "repositories require a newer Git or a fully local "
                             "clone prepared outside this script")
        if config.returncode != 1:
            raise ValueError("Cannot verify repository lazy-fetch configuration")
    result = subprocess.run(
        ["git", "--no-pager", "--no-replace-objects", *offline_args,
         "-c", "core.fsmonitor=false",
         "-C", str(root), *args], input=data, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, timeout=120,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip()
                         or f"git {args[0]} failed ({result.returncode})")
    return result.stdout


class Repo:
    def __init__(self, root, ref):
        self.root = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel"))
                         .removesuffix("\n")).resolve()
        self.sha = git(self.root, "rev-parse", "--verify", "--end-of-options",
                       ref + "^{commit}").decode().strip()
        self.ref = ref
        self.storage_roots = [
            (self.root / os.fsdecode(git(self.root, "rev-parse", option))
             .removesuffix("\n")).resolve()
            for option in ("--git-dir", "--git-common-dir")
        ]
        self.files = {}
        for record in git(self.root, "ls-tree", "-rz", self.sha).split(b"\0"):
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if kind in ("blob", "commit"):
                self.files[path.decode("utf-8", errors="surrogateescape")] = (mode, oid)
        self.source_files = {p for p, (mode, _) in self.files.items()
                             if mode in ("100644", "100755")}
        self.cache = {}

    def prefetch(self, paths):
        needed = sorted(set(paths).intersection(self.source_files).difference(self.cache))
        if not needed:
            return
        payload = "".join(self.files[p][1] + "\n" for p in needed).encode()
        data = git(self.root, "cat-file", "--batch", data=payload)
        offset = 0
        for path in needed:
            end = data.index(b"\n", offset)
            header = data[offset:end].decode().split()
            if len(header) == 2 and header[1] == "missing":
                raise ValueError(f"Missing local Git object for {path}; prepare objects "
                                 "outside this script and retry (lazy fetch disabled)")
            if len(header) != 3:
                raise ValueError("Unexpected Git batch response")
            oid, kind, size = header
            if kind != "blob" or oid != self.files[path][1]:
                raise ValueError("Unexpected Git batch response")
            offset = end + 1
            content = data[offset:offset + int(size)]
            # ast.parse accepts bytes and honors Python encoding cookies/BOMs.
            # Lossy UTF-8 decoding can collapse distinct valid literal values.
            self.cache[path] = content
            offset += int(size) + 1

    def source(self, path):
        if path not in self.source_files:
            raise ValueError(f"Not a regular source file: {path}; links are compared, not followed")
        self.prefetch([path])
        return self.cache[path]

    def module_path(self, module):
        parts = module.split(".")
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            if (parent + ".py" in self.source_files
                    and parent + "/__init__.py" not in self.source_files):
                return None
        stem = module.replace(".", "/")
        return next((p for p in (stem + "/__init__.py", stem + ".py")
                     if p in self.source_files), None)

    def describe(self):
        keys = git(self.root, "config", "--null", "--name-only", "--list").decode().split("\0")
        has_filters = any(key.lower().startswith("filter.") and
                          key.lower().endswith((".clean", ".process")) for key in keys)
        # Status may execute a configured filter while refreshing tracked files.
        if has_filters:
            entries, state = None, "unavailable_configured_filters"
        else:
            status = git(self.root, "status", "--porcelain=v1", "-z",
                         "--untracked-files=normal", "--ignore-submodules=all")
            entries, state = 0, "observed"
            records = iter(status.split(b"\0"))
            for record in records:
                if not record:
                    continue
                entries += 1
                # Porcelain -z emits a second path for renames and copies.
                if b"R" in record[:2] or b"C" in record[:2]:
                    next(records)
        return {"root": str(self.root), "requested_ref": self.ref, "sha": self.sha,
                "working_tree_entries": entries, "working_tree_status": state,
                "submodule_worktrees": "excluded; committed gitlinks compared",
                "input_mode": "committed_tree_only; uncommitted changes excluded"}


def literal(node, values):
    """Resolve a small constant-expression subset; never eval source code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(literal(x, values) for x in node.elts)
    if isinstance(node, ast.JoinedStr):
        # Dropping conversion/formatting can silently select a different module.
        if any(isinstance(x, ast.FormattedValue) and
               (x.conversion != -1 or x.format_spec is not None)
               for x in node.values):
            raise ValueError("Unsupported f-string conversion or format specifier")
        return "".join(str(literal(x.value if isinstance(x, ast.FormattedValue) else x,
                                   values)) for x in node.values)
    if isinstance(node, ast.Subscript):
        return literal(node.value, values)[literal(node.slice, values)]
    raise ValueError("Not a supported literal expression")


def constants(tree):
    def assigned_names(target):
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [name for child in target.elts for name in assigned_names(child)]
        if isinstance(target, ast.Starred):
            return assigned_names(target.value)
        return []

    values = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        # A bare annotation does not assign a value at runtime.
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            if isinstance(node, ast.AugAssign):
                raise ValueError("Augmented assignments require manual review")
            value = literal(node.value, values)
        except (ValueError, KeyError, TypeError, IndexError):
            for target in targets:
                for name in assigned_names(target):
                    values.pop(name, None)
            continue
        for target in targets:
            for name in assigned_names(target):
                values.pop(name, None)
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def find_function(tree, symbol):
    owner = tree
    for name in symbol.split("."):
        if not isinstance(owner, (ast.Module, ast.ClassDef)):
            return None
        matches = [n for n in getattr(owner, "body", [])
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == name]
        if len(matches) != 1:
            return None
        owner = matches[0]
    return owner if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def parameter_names(tree, symbol):
    owner = find_function(tree, symbol)
    if owner is None:
        return None
    args = owner.args
    return [a.arg for a in args.posonlyargs + args.args] + (
        [args.vararg.arg] if args.vararg else []
    ) + [a.arg for a in args.kwonlyargs] + ([args.kwarg.arg] if args.kwarg else [])


def classmethod_binding(function, symbol, expression):
    """Resolve a direct descriptor access; aliases/custom decorators stay unknown."""
    if not any(isinstance(d, ast.Name) and d.id == "classmethod"
               for d in function.decorator_list):
        return None
    if len(function.decorator_list) != 1:
        return "unresolved"
    unbound = isinstance(expression, ast.Attribute) and expression.attr == "__func__"
    if unbound:
        expression = expression.value
    names = []
    while isinstance(expression, ast.Attribute):
        names.append(expression.attr)
        expression = expression.value
    if not isinstance(expression, ast.Name):
        return "unresolved"
    names.append(expression.id)
    parts = symbol.split(".")
    if list(reversed(names))[-len(parts):] != parts:
        return "unresolved"
    if not unbound and not (function.args.posonlyargs or function.args.args
                            or function.args.vararg):
        return "unresolved"
    return "unbound" if unbound else "bound"


def function_contract(node):
    if node is None:
        return None

    def dump(value):
        return ast.dump(value, include_attributes=False) if value is not None else None

    args = node.args
    parameters = []

    def add(argument, kind, default=None):
        parameters.append({"name": argument.arg, "kind": kind,
                           "has_default": default is not None,
                           "default_ast": dump(default), "annotation_ast": dump(argument.annotation)})

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    for i, (argument, default) in enumerate(zip(positional, defaults)):
        add(argument, "positional_only" if i < len(args.posonlyargs) else "positional_or_keyword", default)
    if args.vararg:
        add(args.vararg, "var_positional")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add(argument, "keyword_only", default)
    if args.kwarg:
        add(args.kwarg, "var_keyword")
    body = node.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    normalized_body = json.dumps([dump(n) for n in body], ensure_ascii=True)
    return {"parameters": parameters, "returns_ast": dump(node.returns),
            "decorators_ast": [dump(n) for n in node.decorator_list],
            "type_params_ast": [dump(n) for n in getattr(node, "type_params", [])],
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "body_sha256": hashlib.sha256(normalized_body.encode()).hexdigest()}


def compare_contracts(old_tree, new_tree, symbol):
    old = function_contract(find_function(old_tree, symbol))
    new = function_contract(find_function(new_tree, symbol))
    changed = [key for key in old if old[key] != new[key]] if old and new else []
    return {"status": "symbol_unresolved" if old is None or new is None else
            "changed_review_required" if changed else "unchanged_static_only",
            "changed_fields": changed, "baseline": old, "target": new}


def signature_checks(tree, values, target_tree, module):
    checks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("require_exact_signature", "require_parameter_names"):
            continue
        try:
            symbol = literal(node.args[1], values)
            if node.func.id == "require_parameter_names":
                expected = literal(node.args[2], values)
            else:
                kw = {x.arg: x.value for x in node.keywords}
                expected = literal(kw["positional"], values) if "positional" in kw else ()
                if "var_positional" in kw:
                    name = literal(kw["var_positional"], values)
                    if name is not None:
                        expected += (name,)
                if "keyword_only" in kw:
                    expected += literal(kw["keyword_only"], values)
                if "var_keyword" in kw:
                    name = literal(kw["var_keyword"], values)
                    if name is not None:
                        expected += (name,)
            if not isinstance(symbol, str) or not symbol.startswith(module + "."):
                continue
            relative_symbol = symbol[len(module) + 1:]
            function = find_function(target_tree, relative_symbol)
            binding = classmethod_binding(function, relative_symbol, node.args[0]) if function else None
            actual = parameter_names(target_tree, relative_symbol) if function else None
            if binding == "bound" and (function.args.posonlyargs or function.args.args):
                actual = actual[1:]
            if binding == "unresolved":
                actual = None
            checks.append({"symbol": symbol, "guard_line": node.lineno,
                           "expected_names": list(expected), "target_names": actual,
                           "status": "binding_unresolved" if binding == "unresolved" else
                           "symbol_unresolved" if actual is None else
                           "names_match_only" if list(expected) == actual else
                           "parameter_names_changed"})
        except (ValueError, KeyError, TypeError, IndexError):
            continue
    return checks


def scan_adapters(plugin, fork, target, warnings):
    paths = [p for p in sorted(plugin.source_files)
             if p.startswith("vllm_hcu/patch/") and p.endswith(".py")]
    plugin.prefetch(paths)
    parsed = []
    for path in paths:
        try:
            tree = ast.parse(plugin.source(path))
        except SyntaxError as exc:
            warnings.append(f"AST unavailable: {path}:{exc.lineno}")
            continue
        values = constants(tree)
        module = values.get("TARGET_MODULE")
        if isinstance(module, str) and module.startswith("vllm."):
            parsed.append((path, tree, values, module))
        elif any(isinstance(n, ast.Name) and n.id == "TARGET_MODULE"
                 and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)):
            warnings.append(f"Unresolved TARGET_MODULE: {path}; manual target review required")
    target.prefetch([p for _, _, _, m in parsed if (p := target.module_path(m))])
    fork.prefetch([p for _, _, _, m in parsed if (p := fork.module_path(m))])
    rows = []
    for path, tree, values, module in parsed:
        old_path, new_path = fork.module_path(module), target.module_path(module)
        target_tree = None
        fork_tree = None
        if new_path:
            try:
                target_tree = ast.parse(target.source(new_path))
            except SyntaxError as exc:
                warnings.append(f"AST unavailable: {new_path}:{exc.lineno}")
        if old_path:
            try:
                fork_tree = ast.parse(fork.source(old_path))
            except SyntaxError as exc:
                warnings.append(f"Fork AST unavailable: {old_path}:{exc.lineno}")
        parts = module.split(".")
        prefix = "/".join(parts[:max(1, len(parts) - 2)]) + "/"
        basename = module.split(".")[-1] + ".py"
        candidates = sorted(p for p in target.source_files if p.startswith(prefix)
                            and PurePosixPath(p).name == basename) if not new_path else []
        checks = signature_checks(tree, values, target_tree, module)
        for check in checks:
            check["contract_comparison"] = compare_contracts(
                fork_tree, target_tree, check["symbol"][len(module) + 1:])
        rows.append({"patch_id": values.get("PATCH_ID", path), "adapter": path,
                     "target_module": module, "fork_path": old_path, "official_path": new_path,
                     "module_status": "missing" if not new_path else "unchanged" if old_path
                     and fork.files[old_path] == target.files[new_path] else "changed",
                     "relocation_candidates": candidates,
                     "signature_checks": checks,
                     "upstream_review": "implementation_present_review_semantics" if new_path
                     else "locate_new_owner_or_confirm_removal",
                     "disposition": "pending", "hcu_execution_evidence": None,
                     "correctness_evidence": None, "performance_evidence": None})
    return rows


def fork_deltas(base, fork, target):
    rows = []
    for path in sorted(base.files.keys() | fork.files.keys()):
        b, f, t = base.files.get(path), fork.files.get(path), target.files.get(path)
        if b == f:
            continue
        hint = ("shared_deletion_candidate" if f is None and t is None else
                "identical_file_candidate" if f is not None and f == t else
                "target_unchanged_review_carried_delta" if t == b else
                "target_changed_requires_semantic_review")
        rows.append({"path": path, "change": "added" if b is None else
                     "deleted" if f is None else "modified", "upstream_review": hint,
                     "base_mode": b[0] if b else None,
                     "fork_mode": f[0] if f else None,
                     "target_mode": t[0] if t else None,
                     "disposition": "pending", "hcu_execution_evidence": None,
                     "correctness_evidence": None, "performance_evidence": None})
    return rows


def markdown(report):
    def cell(value):
        # Preserve Git's arbitrary path bytes as visible, reversible escapes.
        value = str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")
        return html.escape(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# 升级只读预检", "", "静态候选清单；所有处置待定。不是兼容性、优化等价或硬件验收报告。", "",
             "## 源码", "", "| 角色 | SHA | 工作区条目 |", "| --- | --- | --- |"]
    for role, source in report["sources"].items():
        entries = source["working_tree_entries"]
        if entries is None:
            entries = "unavailable (configured filters)"
        lines.append(f"| {role} | {source['sha']} | {entries} |")
    lines += ["", "输入仅为提交树；未提交改动不参与分析。", "", "## Fork 携带差异", "",
              "| 文件 | 变化 | 上游核验线索 | 处置 |", "| --- | --- | --- | --- |"]
    for row in report["fork_deltas"]:
        lines.append("| " + " | ".join(cell(row[k]) for k in
                     ("path", "change", "upstream_review", "disposition")) + " |")
    lines += ["", "## 插件挂载点", "", "| Patch | 目标模块 | 模块状态 | 静态契约变化/未解析 | 处置 |",
              "| --- | --- | --- | --- | --- |"]
    for row in report["adapters"]:
        issues = [s["symbol"] + ": " + s["status"] for s in row["signature_checks"]
                  if s["status"] != "names_match_only"]
        issues += [s["symbol"] + ": " + (", ".join(s["contract_comparison"]["changed_fields"])
                   or s["contract_comparison"]["status"]) for s in row["signature_checks"]
                   if s["contract_comparison"]["status"] != "unchanged_static_only"]
        lines.append("| " + " | ".join(cell(x) for x in
                     (row["patch_id"], row["target_module"], row["module_status"],
                      "; ".join(issues) or "仍需审查完整契约", "pending")) + " |")
    lines += ["", "## 覆盖限制", ""] + ["- " + x for x in report["limitations"]]
    lines += ["", "## 解析告警", ""] + ["- " + cell(x) for x in report["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("plugin", "fork", "upstream"):
        parser.add_argument(f"--{role}-root", required=True, type=Path)
    parser.add_argument("--plugin-ref", default="HEAD")
    parser.add_argument("--fork-ref", default="HEAD")
    parser.add_argument("--target-ref", required=True,
                        help="Explicit official tag, branch or commit; no default target")
    parser.add_argument("--upstream-base", required=True, help="Official base commit available in fork history")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside source repositories")
    args = parser.parse_args()
    try:
        plugin = Repo(args.plugin_root, args.plugin_ref)
        fork = Repo(args.fork_root, args.fork_ref)
        target = Repo(args.upstream_root, args.target_ref)
        base = Repo(args.fork_root, args.upstream_base)
        output = args.output.resolve()
        excluded = {path for repo in (plugin, fork, target)
                    for path in (repo.root, *repo.storage_roots)}
        if any(output == path or path in output.parents for path in excluded):
            raise ValueError("Output must be outside all source repositories and Git storage")
        if output.exists():
            raise ValueError("Output directory already exists; choose a new directory")
        git(fork.root, "merge-base", "--is-ancestor", base.sha, fork.sha)
        warnings = []
        report = {"schema_version": 2,
                  "sources": {"plugin": plugin.describe(), "fork": fork.describe(),
                              "target": target.describe(), "official_base": base.describe()},
                  "fork_base_ancestry": "verified; target ancestry not inferred from tree comparison",
                  "fork_deltas": fork_deltas(base, fork, target),
                  "adapters": scan_adapters(plugin, fork, target, warnings),
                  "warnings": warnings,
                  "limitations": [
                      "Fork 差异按路径列项，rename 的旧/新路径分别计算，数量可能不同于 git diff --stat。",
                      "符号链接与 gitlink/submodule 指针参与差异比较，但不跟随链接或递归解析子模块内容；相关依赖另行审查。",
                      "仅扫描可静态解析的 TARGET_MODULE 和部分参数名守卫；不是完整注册清单。",
                      "已解析守卫符号比较 fork 与目标的参数类别/默认值/注解、返回注解、装饰器、async 与函数体 AST；变化是审查线索，不是已证明的 API break。",
                      "动态注册、继承/导入别名、整模块替换、model/quantization 注册、native ABI、全局依赖与完整调用语义仍需审查；未解析不计通过。",
                      "文件相同、符号存在或参数名匹配不证明 HCU 路径可达、优化等价或性能满足要求。",
                      "移除重复补丁须补官方源码/调用链证据及 HCU 正确性、性能和功能矩阵；所有自动处置均保持 pending。",
                      "未 fetch、checkout、安装依赖、导入仓库代码或运行硬件；禁用 lazy fetch，缺历史或对象请在外部准备后重跑。",
                      "配置 clean/process filter 的仓库不执行 git status，工作区条目数记为不可用；提交树分析照常进行。",
                  ]}
        metadata = "vllm_hcu/version.py"
        if metadata in plugin.source_files:
            try:
                report["plugin_declared_metadata"] = constants(ast.parse(plugin.source(metadata)))
            except SyntaxError:
                warnings.append("Plugin version metadata could not be parsed")
        output.mkdir(parents=True, exist_ok=False)
        (output / "preflight.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        (output / "preflight.md").write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"status": "report_generated_not_acceptance", "output": str(output), "fork_deltas": len(report["fork_deltas"]),
                          "adapters": len(report["adapters"]), "warnings": len(warnings)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
