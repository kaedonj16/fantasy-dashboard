"""Make payloads JSON-safe by replacing non-finite floats with None.

json.dumps happily emits NaN/Infinity literals, which are invalid JSON and
make the browser's fetch().json() throw. Consolidates app.py's two former
near-duplicates (_sanitize_for_json handled NaN and infinities;
clean_nan_for_json handled only NaN) into one function that handles both.
"""
import math


def sanitize_for_json(obj):
    """Recursively replace NaN/inf/-inf floats with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
