"""safe_jsmin: rjsmin must never eat spaces inside template literals.

Bare rjsmin.jsmin mangles backtick templates (it stripped the leading space
of a nested template and the space between two interpolations), which
shipped visibly broken UI text ("Week 15· atMIN", "% fair" -> "%fair").
safe_jsmin lifts template literals out before rjsmin runs and restores them
verbatim; these tests pin that contract, the decline paths, and the sidecar
epoch that forces pre-guard .min outputs to be rebuilt.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rjsmin")

from utils import static_minify
from utils.static_minify import (
    _scan_templates,
    _template_texts,
    safe_jsmin,
    sidecar_matches,
    sidecar_value,
)

STATIC = Path(static_minify.STATIC_DIR)


def _ok(src: str) -> str:
    out = safe_jsmin(src)
    assert out is not None, f"safe_jsmin declined: {src!r}"
    assert _template_texts(out) == _template_texts(src)
    return out


def test_playoff_snippet_templates_survive_byte_identical():
    src = (
        "const _left = `Week ${w.week}${w.opp ? ` <span class=\"pm-playoff-loc\">"
        "· ${_loc} ${_pmEsc(w.opp)}</span>` : ''}`;"
    )
    out = _ok(src)
    # The two spaces bare rjsmin provably strips.
    assert "` <span class=\"pm-playoff-loc\">· ${_loc} ${_pmEsc(w.opp)}</span>`" in out
    # Guard the guard: bare rjsmin really does damage this snippet, so the
    # assertions above cannot pass vacuously.
    import rjsmin
    assert "${_loc}${_pmEsc" in rjsmin.jsmin(src)


def test_adjacent_interpolations_keep_their_space():
    out = _ok("const s = `${a} ${b} and ${c}`;")
    assert "`${a} ${b} and ${c}`" in out


def test_template_starting_with_a_space():
    out = _ok("const s = ` hello ${name}`;")
    assert "` hello ${name}`" in out


def test_nested_templates_inside_interpolation():
    out = _ok("const s = `a ${cond ? `b ${deep} c` : `d`} e`;")
    assert "`a ${cond ? `b ${deep} c` : `d`} e`" in out


def test_quotes_and_backticks_inside_strings_are_not_templates():
    src = (
        'const a = "has a ` backtick";\n'
        "const b = 'it\\'s fine';\n"
        "const c = `real ${a}`;"
    )
    out = _ok(src)
    assert _template_texts(src) == ["`real ${a}`"]
    assert "`real ${a}`" in out


def test_comment_with_backtick_is_not_a_template():
    src = "// don't `use` this\nconst s = `real ${x}`;"
    out = _ok(src)
    assert _template_texts(src) == ["`real ${x}`"]
    assert "`real ${x}`" in out


def test_regex_containing_backtick_and_quotes():
    src = "const re = /`'\"/g;\nconst s = `after ${x}`;"
    out = _ok(src)
    assert _template_texts(src) == ["`after ${x}`"]
    assert "`after ${x}`" in out


def test_regex_with_escaped_slash_class_then_template():
    src = "const m = p.match(/\\/([^\\/]+)/);\nconst s = `id ${m[1]} done`;"
    out = _ok(src)
    assert "`id ${m[1]} done`" in out


def test_division_does_not_desync_the_scanner():
    # 'of' in row.of is a property name, not the for-of keyword.
    src = (
        "const q = row.of / 3;\n"
        "const r = a / b / c;\n"
        "if (q / 2 > 1) { t = `q=${q}`; }\n"
        "const u = x / y;"
    )
    out = _ok(src)
    assert "`q=${q}`" in out


def test_template_after_control_flow_paren_is_not_a_tag():
    # if (...) <template statement> -- the ( ... ) is a header, not a call.
    out = _ok("if (ready) `plain ${y}`;")
    assert "`plain ${y}`" in out


def test_tagged_templates_decline():
    assert safe_jsmin("const s = String.raw`a\\n${x}`;") is None
    assert safe_jsmin("const s = tag`hi ${x}`;") is None
    assert safe_jsmin("const s = ns.tag`x`;") is None
    assert safe_jsmin("foo()`x`;") is None


def test_unterminated_constructs_decline():
    assert safe_jsmin("const s = `oops ${x;") is None
    assert safe_jsmin('const s = "oops;') is None
    assert safe_jsmin("const s = /oops;") is None


def test_no_templates_still_minifies():
    out = safe_jsmin("function add(a, b) {\n  // sum\n  return a + b;\n}\n")
    assert out is not None and "return a+b" in out


def test_sidecar_epoch_rejects_pre_guard_values():
    h = "deadbeefcafe"
    assert sidecar_matches(sidecar_value(h), h)
    # A sidecar written by the old bare-hash pipeline must NOT count as
    # fresh, or the mangled .min file it describes would be served forever.
    assert not sidecar_matches(h, h)
    assert not sidecar_matches("", h)
    assert not sidecar_matches(None, h)
    assert not sidecar_matches(sidecar_value("otherhash"), h)


def test_real_player_modal_templates_intact():
    text = (STATIC / "player_modal.js").read_text(encoding="utf-8")
    out = safe_jsmin(text)
    assert out is not None, "safe_jsmin declined player_modal.js"
    assert "` <span class=\"pm-playoff-loc\">· ${_loc} ${_pmEsc(w.opp)}</span>`" in out
    assert _template_texts(out) == _template_texts(text)
    assert len(out) < len(text)


def test_real_app_js_templates_intact():
    text = (STATIC / "app.js").read_text(encoding="utf-8")
    out = safe_jsmin(text)
    assert out is not None, "safe_jsmin declined app.js"
    assert "% fair" in out  # a quasi bare rjsmin collapsed to "%fair"
    assert _template_texts(out) == _template_texts(text)
    assert len(out) < len(text)


def test_served_name_rebuilds_stale_pre_guard_min(monkeypatch, tmp_path):
    import hashlib

    src_text = "const s = `keep ${a} and ${b}`;\nfunction f() { return s; }\n"
    (tmp_path / "widget.js").write_text(src_text, encoding="utf-8")
    src_hash = hashlib.md5(src_text.encode("utf-8")).hexdigest()
    # Simulate a boot under the old pipeline: mangled output + bare-hash
    # sidecar claiming it is current.
    (tmp_path / "widget.min.js").write_text("MANGLED", encoding="utf-8")
    (tmp_path / "widget.min.js.src").write_text(src_hash, encoding="utf-8")

    monkeypatch.setattr(static_minify, "STATIC_DIR", tmp_path)
    static_minify._CACHE.clear()
    try:
        assert static_minify.served_name("widget.js") == "widget.min.js"
        rebuilt = (tmp_path / "widget.min.js").read_text(encoding="utf-8")
        assert rebuilt != "MANGLED"
        assert "`keep ${a} and ${b}`" in rebuilt
        assert sidecar_matches(
            (tmp_path / "widget.min.js.src").read_text(encoding="utf-8"), src_hash)
        # Second call: the epoch sidecar now counts as fresh, no rebuild.
        assert static_minify.served_name("widget.js") == "widget.min.js"
    finally:
        static_minify._CACHE.clear()


def test_served_name_declines_to_original_for_tagged_template(monkeypatch, tmp_path):
    (tmp_path / "tagged.js").write_text(
        "const s = String.raw`a\\n${x}`;\n", encoding="utf-8")
    monkeypatch.setattr(static_minify, "STATIC_DIR", tmp_path)
    static_minify._CACHE.clear()
    try:
        assert static_minify.served_name("tagged.js") == "tagged.js"
        assert not (tmp_path / "tagged.min.js").exists()
    finally:
        static_minify._CACHE.clear()
