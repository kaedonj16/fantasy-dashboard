"""Guard: the dashboard's compute_awards_season() call matches the signature.

build_dashboard_body() calls compute_awards_season(); the render path isn't
unit-tested (it needs a full league ctx + DB), so when the function grew four
required params (#668) the 3-arg call site wasn't updated and every dashboard
load 500'd with a TypeError. This checks the arity statically — no DB needed —
so a future signature change without updating the caller fails here instead of
in production.
"""
import ast
import inspect
import os

from dashboard_services.awards import compute_awards_season

_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_dashboard_calls_awards_with_enough_args():
    sig = inspect.signature(compute_awards_season)
    required = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    n_required = len(required)

    tree = ast.parse(open(_APP, encoding="utf-8").read())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "compute_awards_season"
    ]
    assert calls, "expected a compute_awards_season() call in app.py"
    for c in calls:
        passed = len(c.args) + len(c.keywords)
        has_star = any(isinstance(a, ast.Starred) for a in c.args)
        assert has_star or passed >= n_required, (
            f"compute_awards_season() call at app.py:{c.lineno} passes {passed} "
            f"args but the function requires {n_required} — the dashboard would "
            f"500 with a TypeError"
        )
