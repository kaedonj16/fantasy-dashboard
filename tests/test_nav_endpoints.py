"""Static guard: every endpoint the nav feeds to url_for must actually exist.

build_nav() turns endpoint-name strings into links via url_for(ep, ...), both in
the mobile dock (_NAV_PAGE_META) and the desktop dropdowns (nav_pill_dropdown).
When a route is extracted from app.py into a blueprint its endpoint name gains a
``<blueprint>.`` prefix, so a stale bare name in the nav tables raises a
werkzeug BuildError at render time — a 500 on every page that builds that nav.

The page-rendering integration tests would catch this, but they need the full
Flask/pandas stack and skip in the pytest-only CI job. This test reproduces the
check purely from the AST (no app import), so the whole class of "nav points at
a moved/renamed endpoint" regressions is caught on every push.
"""
import ast
import glob
import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _route_endpoints() -> set[str]:
    """The set of endpoints Flask will register: app.py @app.route handlers by
    function name, and routes/*.py blueprint handlers as ``<bp_name>.<func>``."""
    eps: set[str] = set()

    app_tree = ast.parse(open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read())
    for node in ast.walk(app_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                        and isinstance(d.func.value, ast.Name)
                        and d.func.value.id == "app"
                        and d.func.attr in ("route", "get", "post")):
                    eps.add(node.name)

    for f in glob.glob(os.path.join(_ROOT, "routes", "*.py")):
        tree = ast.parse(open(f, encoding="utf-8").read())
        # <var> = Blueprint("name", ...)
        bpvar_to_name: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if (isinstance(fn, ast.Name) and fn.id == "Blueprint"
                        and node.value.args
                        and isinstance(node.value.args[0], ast.Constant)
                        and isinstance(node.value.args[0].value, str)):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            bpvar_to_name[tgt.id] = node.value.args[0].value
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.decorator_list:
                    if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                            and isinstance(d.func.value, ast.Name)
                            and d.func.value.id in bpvar_to_name
                            and d.func.attr in ("route", "get", "post")):
                        eps.add(f"{bpvar_to_name[d.func.value.id]}.{node.name}")
    return eps


def _url_for_literal_refs() -> set[str]:
    """Endpoint literals passed to url_for("ep", ...) and _section_title_link(
    label, "ep", ...) across app.py and the blueprints. These are just as prone
    to breaking on a route move as the nav tables — and _section_title_link in
    particular swallows a BuildError and silently drops the link, so a stale ref
    wouldn't even 500. Only string literals are checked (dynamic endpoints are
    skipped)."""
    import glob
    refs: set[str] = set()
    files = [os.path.join(_ROOT, "app.py")] + glob.glob(os.path.join(_ROOT, "routes", "*.py"))
    for f in files:
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if (node.func.id == "url_for" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                refs.add(node.args[0].value)
            if (node.func.id == "_section_title_link" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                refs.add(node.args[1].value)
    return {r for r in refs if r}


def _nav_endpoint_refs() -> set[str]:
    """Endpoint strings build_nav hands to url_for: the _NAV_PAGE_META values and
    the item tuples passed to nav_pill_dropdown(...)."""
    tree = ast.parse(open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read())
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_NAV_PAGE_META" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                for v in node.value.values:
                    if (isinstance(v, ast.Tuple) and len(v.elts) >= 2
                            and isinstance(v.elts[1], ast.Constant)):
                        refs.add(v.elts[1].value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "nav_pill_dropdown"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.List)):
            for it in node.args[1].elts:
                if (isinstance(it, ast.Tuple) and len(it.elts) >= 2
                        and isinstance(it.elts[1], ast.Constant)):
                    refs.add(it.elts[1].value)
    return {r for r in refs if isinstance(r, str) and r}


def test_nav_endpoints_all_resolve():
    endpoints = _route_endpoints()
    refs = _nav_endpoint_refs() | _url_for_literal_refs()
    # Sanity: we actually found the tables (guards against the AST scan silently
    # matching nothing if build_nav is refactored).
    assert len(refs) >= 20, f"expected to find the nav endpoint tables, got {len(refs)}"
    missing = sorted(r for r in refs if r not in endpoints)
    assert not missing, (
        "nav references endpoints that no route defines (a moved/renamed route "
        f"whose nav entry wasn't updated, e.g. missing a blueprint prefix): {missing}"
    )
