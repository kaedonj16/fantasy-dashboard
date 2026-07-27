"""Guard: every plain function call in build_dashboard_body matches its def.

The dashboard render path isn't unit-tested (it needs a full league ctx + DB)
and its HTML is cached, so a signature change that isn't mirrored at the call
site stays invisible until the cache expires and the page rebuilds — then it
500s in production with a TypeError. This happened twice from one PR (#668):
compute_awards_season() grew 3->7 params and build_teams_overview() gained a
required `platform`, but the dashboard callers weren't updated.

This checks the arity statically against every first-party def — pure AST, no
imports, so it runs even in the deps-minimal lint job — for the render
functions most exposed to that failure mode.
"""
import ast
import glob
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Render functions with the cache-hides-the-bug exposure. Add more as needed.
_CHECKED_FUNCS = ("build_dashboard_body",)


def _collect_defs():
    """{name: [param-spec]} for every first-party def (skip name collisions later)."""
    defs = {}
    files = (
        [os.path.join(_ROOT, "app.py")]
        + glob.glob(os.path.join(_ROOT, "utils", "**", "*.py"), recursive=True)
        + glob.glob(os.path.join(_ROOT, "dashboard_services", "**", "*.py"), recursive=True)
        + glob.glob(os.path.join(_ROOT, "routes", "**", "*.py"), recursive=True)
    )
    for fp in files:
        try:
            tree = ast.parse(open(fp, encoding="utf-8").read())
        except Exception:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = n.args
            pos = [p.arg for p in a.posonlyargs + a.args]
            required_pos = pos[:-len(a.defaults)] if a.defaults else pos
            required_kwonly = [p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
            defs.setdefault(n.name, []).append({
                "required_pos": required_pos,
                "required_kwonly": required_kwonly,
                "all_pos": pos,
                "varargs": a.vararg is not None,
                "kwargs": a.kwarg is not None,
            })
    return defs


def _mismatches_in(func_node, defs):
    out = []
    for c in ast.walk(func_node):
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)):
            continue
        specs = defs.get(c.func.id)
        if not specs or len(specs) != 1:
            continue  # unknown or ambiguous (same name in >1 file) -> can't be sure
        spec = specs[0]
        if spec["varargs"] or spec["kwargs"]:
            continue  # accepts arbitrary args
        if any(isinstance(a, ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords):
            continue  # *args / **kwargs at the call site -> can't verify statically
        supplied = set(spec["all_pos"][: len(c.args)]) | {k.arg for k in c.keywords}
        missing = [p for p in spec["required_pos"] if p not in supplied]
        missing += [p for p in spec["required_kwonly"] if p not in {k.arg for k in c.keywords}]
        if missing:
            out.append(f"app.py:{c.lineno}  {c.func.id}()  missing: {missing}")
    return out


def test_dashboard_render_calls_match_signatures():
    app_tree = ast.parse(open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read())
    defs = _collect_defs()
    funcs = {
        n.name: n for n in ast.walk(app_tree)
        if isinstance(n, ast.FunctionDef) and n.name in _CHECKED_FUNCS
    }
    for name in _CHECKED_FUNCS:
        assert name in funcs, f"{name} not found in app.py"
    problems = []
    for name, node in funcs.items():
        problems += _mismatches_in(node, defs)
    assert not problems, (
        "call site(s) pass too few args (dashboard would 500 with a TypeError):\n  "
        + "\n  ".join(problems)
    )
