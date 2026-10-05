/* Shared week-range control. Loaded by the core shell before page scripts.
 * Used by Advanced Metrics, player modal, and compare modal.
 *
 * Renders a "Full season" chip plus one chip per week. Tapping week chips
 * multi-selects; the range is min..max of the selection. Tapping "Full season"
 * (or deselecting the last week) resets to the full season, reported as
 * onChange(null, null). Otherwise onChange(minWeek, maxWeek).
 */
(function (global) {
  'use strict';

  function normaliseWeeks(min, max, available) {
    var weeks = (available || []).map(Number).filter(function (w) {
      return Number.isInteger(w) && w >= min && w <= max;
    });
    return Array.from(new Set(weeks)).sort(function (a, b) { return a - b; });
  }

  global._wkBarBuild = function (id, min, max, ws, we, available) {
    min = Number(min); max = Number(max);
    if (max < min) return '';
    var valid = normaliseWeeks(min, max, available);
    var hasAvailability = Array.isArray(available);
    var allowed = hasAvailability ? new Set(valid) : null;
    // Full season when no explicit range, or the range already spans everything.
    var isFull = ws == null || we == null || (Number(ws) <= min && Number(we) >= max);
    var selMin = isFull ? null : Math.max(min, Math.min(max, Number(ws)));
    var selMax = isFull ? null : Math.max(min, Math.min(max, Number(we)));
    var chips = '<button type="button" class="wk-chip wk-chip-full'
      + (isFull ? ' active' : '') + '" data-wk-full="1" aria-pressed="'
      + (isFull ? 'true' : 'false') + '">Full season</button>';
    for (var w = min; w <= max; w++) {
      var unavailable = !!allowed && !allowed.has(w);
      var active = !isFull && w >= selMin && w <= selMax;
      chips += '<button type="button" class="wk-chip'
        + (active ? ' active' : '')
        + (unavailable ? ' wk-chip-unavailable' : '')
        + '" data-week="' + w + '"'
        + (unavailable ? ' disabled aria-disabled="true" title="No data for Week ' + w + '"' : '')
        + ' aria-pressed="' + (active ? 'true' : 'false') + '">'
        + w + '</button>';
    }
    return '<div class="wk-chips" id="' + id + '"'
      + ' data-min="' + min + '" data-max="' + max + '"'
      + ' data-available="' + valid.join(',') + '">'
      + chips + '</div>';
  };

  global._wkBarInit = function (id, onChange) {
    var root = document.getElementById(id);
    if (!root || root.dataset.wkInitialised === '1') return;
    root.dataset.wkInitialised = '1';
    var fullBtn = root.querySelector('[data-wk-full]');
    var weekBtns = Array.from(root.querySelectorAll('.wk-chip[data-week]'));

    function selectedWeeks() {
      return weekBtns
        .filter(function (b) { return b.classList.contains('active') && !b.disabled; })
        .map(function (b) { return Number(b.dataset.week); })
        .sort(function (a, b) { return a - b; });
    }

    function paint() {
      var sel = selectedWeeks();
      var isFull = sel.length === 0;
      if (fullBtn) {
        fullBtn.classList.toggle('active', isFull);
        fullBtn.setAttribute('aria-pressed', isFull ? 'true' : 'false');
      }
      weekBtns.forEach(function (b) {
        var on = b.classList.contains('active') && !b.disabled;
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      return sel;
    }

    function emit() {
      var sel = selectedWeeks();
      if (sel.length === 0) onChange(null, null);
      else onChange(sel[0], sel[sel.length - 1]);
    }

    if (fullBtn) {
      fullBtn.addEventListener('click', function () {
        weekBtns.forEach(function (b) { b.classList.remove('active'); });
        paint();
        emit();
      });
    }
    weekBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        if (btn.disabled) return;
        btn.classList.toggle('active');
        paint();
        emit();
      });
    });
    paint();
  };
})(window);
