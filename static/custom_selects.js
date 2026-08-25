// Site custom <select> dropdown (CSD).
// Shared with app.js so chrome-less pages (cheat-sheet embed iframe) get the
// same styled panel without loading the full app bundle. Keep this IIFE in
// sync with the Custom Select Dropdown block in static/app.js.
// ── Custom Select Dropdown (CSD) ─────────────────────────────────────────────
// Replaces every native <select> with a fully-styled dropdown panel.
// The original select stays in the DOM (CSS-hidden via .csd-wrap>select) so
// existing JS can still read/write .value and fire change events normally.
(function () {
  var _seq = 0;
  var _openWrap = null;
  var ARROW = '<svg width="10" height="6" viewBox="0 0 10 6" fill="none"><path d="M1 1L5 5L9 1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function initOne(sel) {
    if (sel._csdDone || sel.multiple || sel.getAttribute('data-no-custom') !== null) return;
    sel._csdDone = true;

    var wasHidden = sel.style.display === 'none';
    var cs = getComputedStyle(sel); // read before moving

    var wrap = document.createElement('div');
    wrap.className = 'csd-wrap';
    if (wasHidden) wrap.style.display = 'none';
    // Full-width form selects are handled by CSS context rules (.csd-wrap inside
    // form rows); for everything else carry over an explicit min-width.
    var mw = parseFloat(cs.minWidth);
    if (mw > 0) wrap.style.minWidth = mw + 'px';

    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'csd-trigger';
    var listId = 'csd-list-' + (++_seq);
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', listId);
    // Inherit key sizing from original select
    trigger.style.fontWeight = cs.fontWeight;
    trigger.style.borderRadius = cs.borderRadius;
    trigger.style.paddingTop = cs.paddingTop;
    trigger.style.paddingBottom = cs.paddingBottom;
    trigger.style.paddingLeft = cs.paddingLeft;

    var valueEl = document.createElement('span');
    valueEl.className = 'csd-value';
    var arrowEl = document.createElement('span');
    arrowEl.className = 'csd-arrow';
    arrowEl.innerHTML = ARROW;
    trigger.appendChild(valueEl);
    trigger.appendChild(arrowEl);

    var list = document.createElement('div');
    list.id = listId;
    list.className = 'csd-list';
    list.setAttribute('role', 'listbox');
    list.style.display = 'none';

    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(trigger);
    wrap.appendChild(list);
    wrap.appendChild(sel); // CSS keeps select hidden inside wrap

    function rebuild() {
      list.innerHTML = '';
      var cur = sel.value;
      for (var i = 0; i < sel.children.length; i++) {
        var child = sel.children[i];
        if (child.tagName === 'OPTGROUP') {
          var gl = document.createElement('div');
          gl.className = 'csd-group-label';
          gl.textContent = child.label;
          list.appendChild(gl);
          for (var j = 0; j < child.children.length; j++) list.appendChild(mkOpt(child.children[j], cur));
        } else if (child.tagName === 'OPTION') {
          list.appendChild(mkOpt(child, cur));
        }
      }
    }

    function mkOpt(opt, curVal) {
      var el = document.createElement('div');
      el.className = 'csd-option' + (opt.disabled ? ' is-disabled' : '') + (opt.value === curVal ? ' is-selected' : '');
      el.dataset.value = opt.value;
      el.textContent = opt.textContent.trim();
      el.setAttribute('role', 'option');
      return el;
    }

    function syncDisplay() {
      var opt = sel.options[sel.selectedIndex];
      valueEl.textContent = opt ? opt.textContent.trim() : '';
      list.querySelectorAll('.csd-option').forEach(function (el) {
        el.classList.toggle('is-selected', el.dataset.value === sel.value);
      });
    }

    rebuild();
    syncDisplay();

    var isOpen = false;
    var focIdx = -1;

    function getEnabled() { return Array.from(list.querySelectorAll('.csd-option:not(.is-disabled)')); }

    function setFocus(idx) {
      var opts = getEnabled();
      opts.forEach(function (el, i) { el.classList.toggle('is-focused', i === idx); });
      focIdx = idx;
      if (idx >= 0 && opts[idx]) opts[idx].scrollIntoView({ block: 'nearest' });
    }

    // Position the fixed-position list against the trigger's viewport rect so it
    // escapes any overflow:hidden/auto ancestor. Called on open and on
    // scroll/resize while open.
    function positionList() {
      var rect    = trigger.getBoundingClientRect();
      var vp      = window.innerHeight;
      var vw      = window.innerWidth;
      var spaceBelow = vp - rect.bottom - 8;
      var spaceAbove = rect.top - 8;
      var maxH    = 280;

      // Match the trigger width as a floor; let long options grow it wider, but
      // never wider than the viewport.
      list.style.minWidth = rect.width + 'px';
      list.style.maxWidth = (vw - 16) + 'px';
      // Clamp horizontally so the list can't run off either edge — it's
      // position:fixed and lives inside the right-aligned settings menu on
      // mobile, which pushed the league switcher's list off-screen.
      var lw   = list.offsetWidth || rect.width;
      var left = Math.min(rect.left, vw - lw - 8);
      list.style.left = Math.max(8, left) + 'px';

      if (spaceBelow >= Math.min(maxH, 120)) {
        // Enough room below - standard position
        list.style.top    = (rect.bottom + 4) + 'px';
        list.style.bottom = 'auto';
        list.style.maxHeight = Math.min(maxH, spaceBelow) + 'px';
      } else if (spaceAbove > spaceBelow) {
        // More room above - flip upward
        list.style.top    = 'auto';
        list.style.bottom = (vp - rect.top + 4) + 'px';
        list.style.maxHeight = Math.min(maxH, spaceAbove) + 'px';
      } else {
        // Default to below with available space
        list.style.top    = (rect.bottom + 4) + 'px';
        list.style.bottom = 'auto';
        list.style.maxHeight = Math.max(spaceBelow, 80) + 'px';
      }
    }

    function openList() {
      if (isOpen) return;
      if (_openWrap && _openWrap !== wrap) _openWrap._csdClose();
      _openWrap = wrap;
      isOpen = true;
      wrap.classList.add('is-open');
      trigger.setAttribute('aria-expanded', 'true');

      // Reset flip so we can measure correctly
      list.style.top    = '';
      list.style.bottom = '';
      list.style.maxHeight = '';
      list.style.display = 'block';

      positionList();

      requestAnimationFrame(function () { list.classList.add('is-visible'); });
      var opts = getEnabled();
      focIdx = opts.findIndex(function (el) { return el.dataset.value === sel.value; });
      setFocus(focIdx);
    }

    wrap._csdReposition = function () { if (isOpen) positionList(); };

    function closeList() {
      if (!isOpen) return;
      isOpen = false;
      if (_openWrap === wrap) _openWrap = null;
      wrap.classList.remove('is-open');
      trigger.setAttribute('aria-expanded', 'false');
      list.classList.remove('is-visible');
      var _l = list;
      setTimeout(function () { if (!isOpen) _l.style.display = 'none'; }, 150);
    }

    wrap._csdClose = closeList;

    function pick(val) {
      sel.value = val;
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      syncDisplay();
      closeList();
    }

    trigger.addEventListener('click', function (e) { e.stopPropagation(); isOpen ? closeList() : openList(); });

    list.addEventListener('click', function (e) {
      var o = e.target.closest('.csd-option:not(.is-disabled)');
      if (o) pick(o.dataset.value);
    });

    list.addEventListener('mousemove', function (e) {
      var o = e.target.closest('.csd-option:not(.is-disabled)');
      if (o) setFocus(getEnabled().indexOf(o));
    });

    trigger.addEventListener('keydown', function (e) {
      var opts = getEnabled();
      if (e.key === 'ArrowDown') { e.preventDefault(); isOpen ? setFocus(Math.min(focIdx + 1, opts.length - 1)) : openList(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); isOpen ? setFocus(Math.max(focIdx - 1, 0)) : openList(); }
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); isOpen ? (focIdx >= 0 ? pick(opts[focIdx].dataset.value) : closeList()) : openList(); }
      else if (e.key === 'Escape') { closeList(); trigger.focus(); }
      else if (e.key === 'Tab') { closeList(); }
    });

    sel.addEventListener('change', syncDisplay);

    // Re-build when options are populated dynamically (team selects loaded via API)
    new MutationObserver(function () { rebuild(); syncDisplay(); }).observe(sel, { childList: true, subtree: true });

    // Mirror show/hide: when external JS sets sel.style.display, update the wrapper
    new MutationObserver(function () {
      wrap.style.display = (sel.style.display === 'none') ? 'none' : '';
    }).observe(sel, { attributes: true, attributeFilter: ['style'] });
  }

  // One global outside-click handler closes any open dropdown
  document.addEventListener('click', function () { if (_openWrap) _openWrap._csdClose(); });

  // The list is position:fixed, so keep it pinned to its trigger when the page
  // or any scroll container moves (capture:true catches inner scrollers too).
  window.addEventListener('scroll', function () { if (_openWrap && _openWrap._csdReposition) _openWrap._csdReposition(); }, true);
  window.addEventListener('resize', function () { if (_openWrap && _openWrap._csdReposition) _openWrap._csdReposition(); });

  window.initCustomSelects = function (root) {
    (root || document).querySelectorAll('select:not([data-no-custom]):not([multiple])').forEach(initOne);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { window.initCustomSelects(); });
  } else {
    window.initCustomSelects();
  }
}());
