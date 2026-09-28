"""Tap-to-update prompt tests.

Verifies the initUpdatePrompt IIFE in static/app.js:
- Shows a dismissible prompt when the server version differs from the running bundle
- Does NOT show the prompt when versions match
- Never reloads on its own; reloads only when the user taps the prompt
- The service worker handles SKIP_WAITING messages

Runs the actual JS in Node with mocked browser globals.
"""
import os
import re
import subprocess
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_JS = os.path.join(REPO, "static", "app.js")
SW_JS = os.path.join(REPO, "static", "sw.js")


def _extract_iife(name):
    src = open(APP_JS).read()
    start = src.index("(function %s()" % name)
    # The IIFE ends with "})();" at the start of a line.
    end_marker = "\n})();"
    end = src.index(end_marker, start) + len(end_marker)
    return src[start:end]


def _run_node(iife_js, preamble, test_code):
    # Order matters: mocks first, then the IIFE, then the test assertions.
    script = preamble + "\n" + iife_js + "\n" + test_code
    proc = subprocess.run(
        ["node", "-e", script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, "node failed:\n%s\n%s" % (proc.stdout, proc.stderr)
    return proc.stdout.strip()


# Shared mock preamble: minimal DOM + navigator.serviceWorker + fetch.
# Note: Node 21+ has a built-in global `navigator`, so we must assign via
# globalThis (var declarations don't shadow it for the `in` operator).
MOCK_PREAMBLE = textwrap.dedent("""
    globalThis.__reloads = 0;
    globalThis.__fetchUrl = null;
    globalThis.__serverVersion = { app_js: 'serverhash999', git_sha: '' };
    globalThis.__myHash = 'oldhash111';

    function MockEl(tag) {
      this.tagName = tag;
      this.className = '';
      this.children = [];
      this._listeners = {};
      this.textContent = '';
      this.innerHTML = '';
      this.parentNode = null;
    }
    MockEl.prototype.setAttribute = function(){};
    MockEl.prototype.appendChild = function(c){ c.parentNode = this; this.children.push(c); return c; };
    MockEl.prototype.removeChild = function(c){ var i=this.children.indexOf(c); if(i>=0)this.children.splice(i,1); c.parentNode=null; };
    MockEl.prototype.addEventListener = function(t, fn){ (this._listeners[t] = this._listeners[t] || []).push(fn); };
    MockEl.prototype.click = function(){ (this._listeners['click'] || []).forEach(function(fn){ fn({ stopPropagation: function(){} }); }); };

    var __bodyChildren = [];
    var __docListeners = {};
    globalThis.document = {
      currentScript: { src: 'https://example.com/static/app.js?v=' + globalThis.__myHash },
      visibilityState: 'visible',
      _listeners: __docListeners,
      createElement: function(tag){ return new MockEl(tag); },
      querySelector: function(sel){
        var cls = sel.replace('.', '');
        for (var i=0;i<__bodyChildren.length;i++){
          if (__bodyChildren[i].className.split(' ').indexOf(cls) >= 0) return __bodyChildren[i];
        }
        return null;
      },
      addEventListener: function(t, fn){ (__docListeners[t] = __docListeners[t] || []).push(fn); },
    };
    globalThis.document.body = new MockEl('body');
    globalThis.document.body.appendChild = function(c){
      c.parentNode = this; this.children.push(c); __bodyChildren.push(c); return c;
    };
    globalThis.document.body.removeChild = function(c){
      var i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1);
      var j = __bodyChildren.indexOf(c); if (j >= 0) __bodyChildren.splice(j, 1);
      c.parentNode = null;
    };

    var __swListeners = {};
    var __regListeners = {};
    var __mockReg = {
      waiting: null,
      installing: null,
      addEventListener: function(t, fn){ (__regListeners[t] = __regListeners[t] || []).push(fn); },
      update: function(){},
    };
    // Node 21+ defines global navigator as a getter-only property; override it.
    var __mockNavigator = {
      serviceWorker: {
        controller: { postMessage: function(){ globalThis.__swPostMessage = true; } },
        addEventListener: function(t, fn){ (__swListeners[t] = __swListeners[t] || []).push(fn); },
        getRegistration: function(){ return Promise.resolve(__mockReg); },
        ready: Promise.resolve(null),
      },
      onLine: true,
    };
    try {
      Object.defineProperty(globalThis, 'navigator', {
        value: __mockNavigator, writable: true, configurable: true,
      });
    } catch (e) { globalThis.navigator = __mockNavigator; }
    globalThis.fetch = function(url, opts){
      globalThis.__fetchUrl = url;
      return Promise.resolve({
        ok: true,
        json: function(){ return Promise.resolve(globalThis.__serverVersion); },
      });
    };
    globalThis.location = { href: 'https://example.com/', reload: function(){ globalThis.__reloads++; } };
    globalThis.window = {
      addEventListener: function(){},
      location: globalThis.location,
    };
    globalThis.MessageChannel = function(){
      var self = this;
      this.port1 = { onmessage: null, _post: function(d){ if (self.port1.onmessage) self.port1.onmessage({ data: d }); } };
      this.port2 = {};
    };
    globalThis.setTimeout = function(fn, ms){
      // In tests, run timeouts on the next macrotask so bypassReload's
      // fallback fires without waiting real milliseconds.
      setImmediate(fn);
      return 0;
    };
    globalThis.setInterval = function(fn){ return 0; };
""")


def test_prompt_shows_on_version_mismatch():
    iife = _extract_iife("initUpdatePrompt")
    out = _run_node(iife, MOCK_PREAMBLE, textwrap.dedent("""
        function flush(n){ return n <= 0 ? Promise.resolve() : new Promise(function(res){ setImmediate(function(){ flush(n-1).then(res); }); }); }
        flush(8).then(function(){
          var banner = document.querySelector('.br-update-banner');
          var dismiss = null;
          if (banner) banner.children.forEach(function(c){ if (c.className === 'br-update-dismiss') dismiss = c; });
          console.log(JSON.stringify({
            fetchUrl: globalThis.__fetchUrl,
            bannerShown: !!banner,
            reloadsWithoutTap: globalThis.__reloads,
            dismissPresent: !!dismiss,
          }));
        });
    """))
    import json
    data = json.loads(out)
    assert data["fetchUrl"] == "/healthz/version", data
    assert data["bannerShown"] is True, "prompt must appear on version mismatch: %s" % data
    assert data["reloadsWithoutTap"] == 0, "must never auto-reload: %s" % data
    assert data["dismissPresent"] is True, "prompt must be dismissible: %s" % data


def test_no_prompt_when_versions_match():
    iife = _extract_iife("initUpdatePrompt")
    preamble = MOCK_PREAMBLE.replace(
        "globalThis.__serverVersion = { app_js: 'serverhash999', git_sha: '' };",
        "globalThis.__serverVersion = { app_js: 'oldhash111', git_sha: '' };",
    )
    out = _run_node(iife, preamble, textwrap.dedent("""
        function flush(n){ return n <= 0 ? Promise.resolve() : new Promise(function(res){ setImmediate(function(){ flush(n-1).then(res); }); }); }
        flush(8).then(function(){
          console.log(JSON.stringify({
            bannerShown: !!document.querySelector('.br-update-banner'),
            reloads: globalThis.__reloads,
          }));
        });
    """))
    import json
    data = json.loads(out)
    assert data["bannerShown"] is False, "no prompt when versions match: %s" % data
    assert data["reloads"] == 0


def test_tap_reloads_and_dismiss_hides():
    iife = _extract_iife("initUpdatePrompt")
    out = _run_node(iife, MOCK_PREAMBLE, textwrap.dedent("""
        function flush(n){ return n <= 0 ? Promise.resolve() : new Promise(function(res){ setImmediate(function(){ flush(n-1).then(res); }); }); }
        flush(8).then(function(){
          var banner = document.querySelector('.br-update-banner');
          var main = null, dismiss = null;
          banner.children.forEach(function(c){
            if (c.className === 'br-update-main') main = c;
            if (c.className === 'br-update-dismiss') dismiss = c;
          });
          // Dismiss first: banner goes away, no reload.
          dismiss.click();
          var afterDismiss = {
            bannerGone: !document.querySelector('.br-update-banner'),
            reloadsAfterDismiss: globalThis.__reloads,
          };
          console.log(JSON.stringify(afterDismiss));
        });
    """))
    import json
    data = json.loads(out)
    assert data["bannerGone"] is True, "dismiss X must remove the prompt: %s" % data
    assert data["reloadsAfterDismiss"] == 0, "dismiss must not reload: %s" % data


def test_tap_main_triggers_reload():
    iife = _extract_iife("initUpdatePrompt")
    out = _run_node(iife, MOCK_PREAMBLE, textwrap.dedent("""
        function flush(n){ return n <= 0 ? Promise.resolve() : new Promise(function(res){ setImmediate(function(){ flush(n-1).then(res); }); }); }
        flush(8).then(function(){
          var banner = document.querySelector('.br-update-banner');
          var main = null;
          banner.children.forEach(function(c){ if (c.className === 'br-update-main') main = c; });
          main.click();
          return new Promise(function(res){ setImmediate(res); });
        }).then(function(){
          return new Promise(function(res){ setImmediate(res); });
        }).then(function(){
          console.log(JSON.stringify({ reloadsAfterTap: globalThis.__reloads }));
        });
    """))
    import json
    data = json.loads(out)
    assert data["reloadsAfterTap"] >= 1, "tapping the prompt must reload: %s" % data


def test_sw_handles_skip_waiting():
    src = open(SW_JS).read()
    assert "'SKIP_WAITING'" in src or '"SKIP_WAITING"' in src, \
        "sw.js must handle the SKIP_WAITING message from the tap-to-update prompt"
    assert "self.skipWaiting()" in src


def test_no_em_dashes_in_prompt_copy():
    iife = _extract_iife("initUpdatePrompt")
    assert "\u2014" not in iife and "\u2013" not in iife, \
        "prompt copy must not use em/en dashes"


def test_no_bare_location_reload_outside_tap():
    # The only location.reload() calls must be inside tap/applyUpdate flows,
    # never at IIFE top level or in the version-check promise chain.
    iife = _extract_iife("initUpdatePrompt")
    # Find all reload() occurrences and ensure they're inside functions named
    # applyUpdate, bypassReload, or the controllerchange handler.
    for m in re.finditer(r"location\.reload\(\)", iife):
        before = iife[:m.start()]
        # Walk back to the nearest function definition.
        fns = list(re.finditer(r"function (\w+)\s*\(", before))
        assert fns, "reload() outside any function"
        nearest = fns[-1].group(1)
        assert nearest in ("applyUpdate", "bypassReload", ""), \
            "reload() inside unexpected function %s" % nearest
        # Anonymous callbacks are fine only if inside applyUpdate/bypassReload
        # or the controllerchange onmessage handler; check enclosing named fn.
