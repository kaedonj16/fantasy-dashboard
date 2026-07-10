"""Guards utils.push_notifications against the psycopg dict_row bug class.

get_conn() uses psycopg's dict_row factory, so query rows are dicts keyed by
column name. The push helpers used to read them positionally (r[0], r[3], tuple
unpacking), which raised KeyError / bound column-name strings and, because the
broadcast callers swallow exceptions, silently sent notifications to nobody.
These pin the column-name access so that can't regress.

Pure unit tests - the module's heavy deps (pywebpush, cryptography) are imported
lazily, so importing these helpers needs nothing extra.
"""
from utils.push_notifications import _filter_prefs


def _row(endpoint, p256dh, auth, prefs=None):
    # Shape of a psycopg dict_row from the push_subscriptions SELECTs.
    return {"endpoint": endpoint, "p256dh": p256dh, "auth": auth,
            "prefs": prefs, "owner_id": "o1"}


def test_filter_prefs_reads_dict_rows_without_notif_type():
    rows = [_row("https://e/1", "k1", "a1"), _row("https://e/2", "k2", "a2")]
    out = _filter_prefs(rows, None)
    assert out == [("https://e/1", "k1", "a1"), ("https://e/2", "k2", "a2")]


def test_filter_prefs_respects_disabled_type():
    import json
    rows = [
        _row("https://on", "k", "a", prefs=json.dumps({"lineup_lock": True})),
        _row("https://off", "k", "a", prefs=json.dumps({"lineup_lock": False})),
        _row("https://default", "k", "a", prefs=None),  # unset -> enabled
    ]
    out = _filter_prefs(rows, "lineup_lock")
    endpoints = [t[0] for t in out]
    assert "https://on" in endpoints
    assert "https://default" in endpoints
    assert "https://off" not in endpoints


def test_filter_prefs_accepts_predecoded_jsonb_prefs():
    # dict_row returns JSONB already decoded to a dict, not a string.
    rows = [_row("https://off", "k", "a", prefs={"waiver": False})]
    assert _filter_prefs(rows, "waiver") == []
    assert _filter_prefs(rows, "trade") == [("https://off", "k", "a")]
