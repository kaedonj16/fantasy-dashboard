"""Gunicorn server hooks for the Render web service.

Loaded via ``--config`` from ``data_building/updates/startup.py`` (the
production entry point). CLI flags in startup.py still take precedence over
anything set here; this file only adds lifecycle hooks.

``post_fork`` fires in each worker AFTER the master has bound the port, so
spawning the deploy-time cache warmup here can never delay or break the
port bind (the failure mode that took down deploys in #1882). The warmup
runs in a daemon thread: requests are served immediately, cold or warm,
and the worker fills its shared caches in the background.
"""


def post_fork(server, worker):
    """Warm league-independent shared caches in this worker, non-blocking."""
    try:
        from dashboard_services.startup_warmup import warm_shared_caches_async
        warm_shared_caches_async()
    except Exception as exc:  # never let a hook break worker boot
        server.log.warning("[startup-warmup] post_fork hook failed: %s", exc)


def when_ready(server):
    server.log.info(
        "[startup-warmup] gunicorn ready; per-worker cache warmup "
        "runs post-fork (STARTUP_WARMUP_ENABLED=0 to disable)"
    )
