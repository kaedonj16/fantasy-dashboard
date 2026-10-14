"""Boot-time minify of first-party static JS/CSS.

Same pattern as app.js / dashboard.css: write a sibling ``*.min.*`` when
rjsmin/rcssmin succeed, keyed by a ``.src`` hash sidecar so the next boot is a
no-op until the source changes. Callers serve whatever ``served_name`` returns.

JS is minified through :func:`safe_jsmin`, never bare ``rjsmin.jsmin``:
rjsmin (still 1.3.0, the latest) mangles template literals -- it strips
significant spaces inside them (the leading space of a nested template and
the space between two ``${...}`` interpolations), which shipped visibly
broken UI text ("Week 15· atMIN", "% fair" -> "%fair"). safe_jsmin lifts
every template literal out before rjsmin runs and restores it verbatim.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

_CACHE: dict[str, str] = {}

# Sidecar epoch for JS outputs. The .src sidecar stores
# ``"<source md5>:<epoch>"`` for JS so a boot after a minify-pipeline change
# regenerates outputs even when the source hash is unchanged (the pre-guard
# rjsmin outputs are mangled and must never be treated as fresh). Bump this
# whenever safe_jsmin's behavior changes. CSS sidecars keep the bare hash:
# the CSS pipeline is unchanged.
MINIFY_EPOCH = "tplguard1"


def sidecar_value(src_hash: str) -> str:
    """Sidecar content for a JS minify output built from ``src_hash``."""
    return f"{src_hash}:{MINIFY_EPOCH}"


def sidecar_matches(stored: str | None, src_hash: str) -> bool:
    """True when a JS .src sidecar holds the current value for ``src_hash``."""
    return (stored or "").strip() == sidecar_value(src_hash)


def static_hash(filename: str) -> str:
    """8-char MD5 of a file under ``static/``, or ``"0"`` if it is missing."""
    path = STATIC_DIR / filename
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


# ── Template-literal-safe JS minification ─────────────────────────────────
#
# rjsmin treats backtick template literals as ordinary code and eats
# significant whitespace inside them. safe_jsmin therefore:
#   1. scans the source with a small JS-aware state machine and records the
#      spans of every OUTERMOST template literal,
#   2. swaps each one for a plain string placeholder,
#   3. runs rjsmin on the protected source (its string handling is sound),
#   4. restores the original template literals verbatim,
#   5. re-scans the result and refuses (returns None) unless the template
#      sequence is byte-identical to the source's.
# The caller serves the original file whenever safe_jsmin returns None:
# correctness over bytes, always.


class _ScanError(Exception):
    """The scanner lost track of the source; the caller must decline."""


# Keywords after which an expression (so a regex, or a plain template
# literal) may follow. Any other identifier before a backtick is a tag.
_EXPR_KEYWORDS = frozenset({
    "await", "case", "default", "delete", "do", "else", "extends", "in",
    "instanceof", "new", "of", "return", "throw", "typeof", "void", "yield",
})
_LITERAL_WORDS = frozenset({"false", "null", "true"})
# Keywords whose (...) is a control-flow header, never a tag call.
_CONTROL_PAREN_WORDS = frozenset({"catch", "for", "if", "switch", "while", "with"})

_PUNCTS = (
    ">>>=", "...", "===", "!==", ">>>", "<<=", ">>=", "**=", "&&=", "||=",
    "??=", "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**",
)


class _TemplateScanner:
    """Find the (start, end) spans of every outermost template literal.

    Tracks just enough JS structure to stay in sync: quoted strings, regex
    literals (previous-significant-token heuristic), comments, and
    ``${...}`` nesting. ``tagged`` is set when any template literal is used
    as a tagged template (its tag expression would be orphaned if the
    literal were swapped for a placeholder string).
    """

    def __init__(self, text: str):
        self.s = text
        self.n = len(text)
        self.templates: list[tuple[int, int]] = []
        self.tagged = False

    def scan(self) -> tuple[list[tuple[int, int]], bool]:
        i = 0
        if self.s.startswith("#!"):  # hashbang line
            nl = self.s.find("\n")
            i = self.n if nl == -1 else nl + 1
        self._code(i, stop_at_brace=False, collect=True, prev=None)
        return self.templates, self.tagged

    # -- token scanners (return the index just past the token) -------------

    def _string(self, i: int) -> int:
        s, n = self.s, self.n
        quote = s[i]
        i += 1
        while i < n:
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == quote:
                return i + 1
            if c == "\n" or c == "\r":
                break
            i += 1
        raise _ScanError("unterminated string literal")

    def _regex(self, i: int) -> int:
        s, n = self.s, self.n
        i += 1  # opening /
        in_class = False
        class_fresh = False
        while i < n:
            c = s[i]
            if c == "\\":
                # An escaped char still counts as class content, so a ']'
                # right after it (e.g. [\/] or [\]]) closes the class.
                class_fresh = False
                i += 2
                continue
            if c == "\n" or c == "\r":
                break
            if in_class:
                if c == "]" and not class_fresh:
                    in_class = False
                class_fresh = False
            elif c == "[":
                in_class = True
                class_fresh = True
                if i + 1 < n and s[i + 1] == "^":
                    i += 1
            elif c == "/":
                i += 1
                while i < n and (s[i].isalnum() or s[i] in "_$"):
                    i += 1  # flags
                return i
            i += 1
        raise _ScanError("unterminated regex literal")

    def _number(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n and (s[i].isalnum() or s[i] in "._"):
            i += 1
        return i

    def _ident_end(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n and (s[i].isalnum() or s[i] in "_$" or ord(s[i]) >= 128):
            i += 1
        return i

    def _template(self, i: int) -> int:
        """Consume a template literal starting at the backtick s[i]."""
        s, n = self.s, self.n
        i += 1
        while i < n:
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == "`":
                return i + 1
            if c == "$" and i + 1 < n and s[i + 1] == "{":
                i = self._code(i + 2, stop_at_brace=True, collect=False,
                               prev=("punct", "${"))
                continue
            i += 1
        raise _ScanError("unterminated template literal")

    def _skip_ws_comments(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n:
            c = s[i]
            if c in " \t\r\n\v\f\u00a0\ufeff\u2028\u2029":
                i += 1
            elif c == "/" and i + 1 < n and s[i + 1] == "/":
                nl = s.find("\n", i + 2)
                i = n if nl == -1 else nl + 1
            elif c == "/" and i + 1 < n and s[i + 1] == "*":
                end = s.find("*/", i + 2)
                if end == -1:
                    raise _ScanError("unterminated block comment")
                i = end + 2
            elif c == "<" and s.startswith("<!--", i):
                nl = s.find("\n", i + 4)
                i = n if nl == -1 else nl + 1
            else:
                break
        return i

    # -- classification helpers ---------------------------------------------

    @staticmethod
    def _regex_allowed(prev) -> bool:
        if prev is None:
            return True
        kind, val = prev
        if kind == "keyword":
            return True
        if kind == "punct":
            return val not in (")", "]", "}", "++", "--")
        return False

    def _is_tagged(self, prev, history) -> bool:
        """True when a backtick right after ``prev`` forms a tagged template."""
        if prev is None:
            return False
        kind, val = prev
        if kind == "ident" or kind == "template":
            return True
        if kind == "punct" and val == "]":
            return True
        if kind == "punct" and val == ")":
            # Find the matching '(' in this token run. Control-flow headers
            # (if/for/while/switch/catch/with) end statements, so a template
            # after them is a plain expression statement, not a tag call.
            depth = 0
            idx = len(history) - 1
            while idx >= 0:
                k, v = history[idx]
                if k == "punct" and v == ")":
                    depth += 1
                elif k == "punct" and v == "(":
                    depth -= 1
                    if depth == 0:
                        break
                idx -= 1
            if idx < 0:
                return True  # no local match: be conservative
            before = history[idx - 1] if idx > 0 else None
            if before is not None and before[1] in _CONTROL_PAREN_WORDS:
                return False
            return True
        return False

    def _line_is_clean(self, i: int) -> bool:
        """True when only whitespace precedes position i on its line."""
        j = i - 1
        while j >= 0 and self.s[j] != "\n":
            if self.s[j] not in " \t\r":
                return False
            j -= 1
        return True

    def _punct(self, i: int) -> str:
        s = self.s
        for p in _PUNCTS:
            if s.startswith(p, i):
                if p == "?." and i + 2 < self.n and s[i + 2].isdigit():
                    return "?"  # ternary ? followed by a .5-style number
                return p
        return s[i]

    # -- main code scanner ----------------------------------------------------

    def _code(self, i: int, *, stop_at_brace: bool, collect: bool, prev) -> int:
        s, n = self.s, self.n
        history: list[tuple[str, str | None]] = []
        if prev is not None:
            history.append(prev)
        depth = 0
        while True:
            i = self._skip_ws_comments(i)
            if i >= n:
                if stop_at_brace:
                    raise _ScanError("unterminated ${...} interpolation")
                return i
            c = s[i]
            if c == "}":
                if stop_at_brace and depth == 0:
                    return i + 1
                if stop_at_brace:
                    depth -= 1
                prev = ("punct", "}")
                history.append(prev)
                i += 1
                continue
            if c == "{":
                if stop_at_brace:
                    depth += 1
                prev = ("punct", "{")
                history.append(prev)
                i += 1
                continue
            if c == '"' or c == "'":
                i = self._string(i)
                prev = ("string", None)
                history.append(prev)
                continue
            if c == "`":
                if collect and self._is_tagged(prev, history):
                    self.tagged = True
                start = i
                i = self._template(i)
                if collect:
                    self.templates.append((start, i))
                prev = ("template", None)
                history.append(prev)
                continue
            if c == "/":
                if self._regex_allowed(prev):
                    i = self._regex(i)
                    prev = ("regex", None)
                elif i + 1 < n and s[i + 1] == "=":
                    prev = ("punct", "/=")
                    i += 2
                else:
                    prev = ("punct", "/")
                    i += 1
                history.append(prev)
                continue
            if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
                i = self._number(i)
                prev = ("number", None)
                history.append(prev)
                continue
            if c.isalpha() or c in "_$" or ord(c) >= 128:
                j = self._ident_end(i)
                word = s[i:j]
                i = j
                # A word right after '.' / '?.' is a property name, never a
                # keyword: row.of / 3 is division, not a regex after 'of'.
                is_property = bool(history) and history[-1] in (
                    ("punct", "."), ("punct", "?."))
                if word in _EXPR_KEYWORDS and not is_property:
                    prev = ("keyword", word)
                elif word in _LITERAL_WORDS and not is_property:
                    prev = ("literal", word)
                else:
                    prev = ("ident", word)
                history.append(prev)
                continue
            if c == "-" and s.startswith("-->", i) and self._line_is_clean(i):
                nl = s.find("\n", i + 3)  # Annex B HTML close comment
                i = n if nl == -1 else nl + 1
                continue
            tok = self._punct(i)
            i += len(tok)
            prev = ("punct", tok)
            history.append(prev)


def _scan_templates(text: str) -> tuple[list[tuple[int, int]], bool]:
    """(outermost template spans, any tagged template?) for JS source."""
    return _TemplateScanner(text).scan()


def _template_texts(text: str) -> list[str]:
    spans, _tagged = _scan_templates(text)
    return [text[a:b] for a, b in spans]


def safe_jsmin(text: str) -> str | None:
    """Minify JS without letting rjsmin near template-literal contents.

    Returns the minified source, or None whenever the result cannot be
    proven template-identical to the input (tagged templates, scanner
    desync, placeholder loss, rjsmin unavailable). None means: serve the
    original file unminified.
    """
    if not isinstance(text, str) or not text:
        return None
    try:
        import rjsmin
    except Exception:
        return None
    try:
        spans, tagged = _scan_templates(text)
        if tagged:
            return None  # a placeholder string would orphan the tag
        if not spans:
            out = rjsmin.jsmin(text)
            return out if out else None
        prefix = "__BR_TPL_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10] + "_"
        if prefix in text:
            return None
        placeholders = [f'"{prefix}{k}__"' for k in range(len(spans))]
        parts: list[str] = []
        last = 0
        for k, (a, b) in enumerate(spans):
            parts.append(text[last:a])
            parts.append(placeholders[k])
            last = b
        parts.append(text[last:])
        minified = rjsmin.jsmin("".join(parts))
        if not minified:
            return None
        for ph in placeholders:
            if minified.count(ph) != 1:
                return None  # rjsmin dropped or duplicated a placeholder
        out = minified
        for k, (a, b) in enumerate(spans):
            out = out.replace(placeholders[k], text[a:b], 1)
        if prefix in out:
            return None  # a placeholder survived restoration
        out_spans, out_tagged = _scan_templates(out)
        if out_tagged or len(out_spans) != len(spans):
            return None
        if [out[a:b] for a, b in out_spans] != [text[a:b] for a, b in spans]:
            return None
        return out
    except Exception:
        return None


def served_name(filename: str) -> str:
    """Filename under ``/static/`` to serve (minified when minify succeeded)."""
    cached = _CACHE.get(filename)
    if cached is not None:
        return cached
    out = _minify(filename)
    _CACHE[filename] = out
    return out


def _minify(filename: str) -> str:
    src = STATIC_DIR / filename
    if not src.exists():
        return filename
    suffix = src.suffix
    if suffix not in (".js", ".css"):
        return filename
    out_name = f"{src.stem}.min{suffix}"
    out = STATIC_DIR / out_name
    meta = STATIC_DIR / f"{out_name}.src"
    try:
        src_bytes = src.read_bytes()
        src_hash = hashlib.md5(src_bytes).hexdigest()
    except OSError:
        return filename
    is_js = suffix == ".js"
    try:
        if out.exists() and meta.exists():
            stored = meta.read_text()
            # JS sidecars carry the minify epoch, so outputs built by an
            # older (template-mangling) pipeline never count as fresh.
            fresh = sidecar_matches(stored, src_hash) if is_js \
                else stored.strip() == src_hash
            if fresh:
                return out_name
    except OSError:
        pass
    text = src_bytes.decode("utf-8")
    # Guard against a broken minifier emitting near-empty output. Typical
    # rjsmin/rcssmin shrink is 20–40%; anything below 40%/30% of source is
    # treated as a failure and we keep the original.
    min_ratio = 0.4 if is_js else 0.3
    try:
        if is_js:
            minified = safe_jsmin(text)
            if minified is None:
                logger.info(
                    "[%s] safe minify declined (template guard), serving original",
                    filename,
                )
                return filename
        else:
            import rcssmin
            minified = rcssmin.cssmin(text)
        if not minified or len(minified) < len(text) * min_ratio:
            logger.info("[%s] minify sanity check failed, serving original", filename)
            return filename
        out.write_text(minified, encoding="utf-8")
        meta.write_text(sidecar_value(src_hash) if is_js else src_hash,
                        encoding="utf-8")
        logger.info(
            "[%s] minified: %d KB -> %d KB",
            filename, len(text) // 1024, len(minified) // 1024,
        )
        return out_name
    except Exception as exc:
        logger.info("[%s] minify unavailable, serving unminified: %s", filename, exc)
        return filename
