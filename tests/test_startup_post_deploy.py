"""The production startCommand runs data_building/updates/startup.py, which must
spawn scripts/post_deploy.py from the repo root — not a path relative to the
updates/ directory, which does not contain that script."""

import ast
import os

_UPDATES_STARTUP = os.path.join(
    os.path.dirname(__file__), "..", "data_building", "updates", "startup.py"
)
_POST_DEPLOY = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "post_deploy.py"
)


def test_post_deploy_script_exists_at_repo_root():
    assert os.path.isfile(_POST_DEPLOY), _POST_DEPLOY


def test_startup_resolves_post_deploy_from_repo_root():
    # Guard against the regression that joined "scripts/post_deploy.py" onto
    # data_building/updates/ (a path that never exists) and silently skipped
    # the deploy-time ADP refresh.
    src = open(os.path.normpath(_UPDATES_STARTUP)).read()
    ast.parse(src)  # still valid Python
    assert "post_deploy.py" in src
    assert '"scripts", "post_deploy.py"' in src or "'scripts', 'post_deploy.py'" in src
    # Must walk up to the repo root, not look in updates/scripts/.
    assert ".." in src
    assert os.path.isfile(os.path.normpath(_POST_DEPLOY))
    # Sanity: the wrong path really does not exist.
    wrong = os.path.join(
        os.path.dirname(os.path.normpath(_UPDATES_STARTUP)), "scripts", "post_deploy.py"
    )
    assert not os.path.exists(wrong), wrong
    # gunicorn must start whether or not post_deploy.py was found.
    assert src.index("os.execvp") > src.index("post_deploy_script")
