/* Shared week-range control. Loaded by the core shell before page scripts.
 * Used by Advanced Metrics, player modal, and compare modal. */
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
    var allowed = new Set(valid);
    var n = max - min + 1;
    var loW = ws != null ? Math.max(min, Math.min(max, Number(ws))) : min;
    var hiW = we != null ? Math.max(min, Math.min(max, Number(we))) : max;
    if (loW > hiW) { var swap = loW; loW = hiW; hiW = swap; }
    var ticks = '';
    for (var w = min; w <= max; w++) {
      var unavailable = hasAvailability && !allowed.has(w);
      ticks += '<span class="wk-tick' + (w >= loW && w <= hiW ? ' wk-tick-in' : '')
        + (unavailable ? ' wk-tick-unavailable' : '') + '" data-week="' + w + '"'
        + (unavailable ? ' aria-disabled="true" title="No data for Week ' + w + '"' : '') + '>W' + w + '</span>';
    }
    var pctL = ((loW - min) / n * 100).toFixed(2);
    var pctR = ((max - hiW) / n * 100).toFixed(2);
    return '<div class="wk-bar' + (n > 10 ? ' wk-bar-dense' : '') + '" id="' + id
      + '" data-min="' + min + '" data-max="' + max + '" data-ws="' + loW + '" data-we="' + hiW
      + '" data-available="' + valid.join(',') + '"><div class="wk-bar-track"><div class="wk-bar-bg"></div>'
      + '<div class="wk-bar-sel" style="left:' + pctL + '%;right:' + pctR + '%">'
      + '<div class="wk-bar-grip wk-bar-grip-l" role="slider" aria-label="Start week" aria-valuemin="' + min
      + '" aria-valuemax="' + max + '" aria-valuenow="' + loW + '" tabindex="0"><span></span><span></span></div>'
      + '<div class="wk-bar-grip wk-bar-grip-r" role="slider" aria-label="End week" aria-valuemin="' + min
      + '" aria-valuemax="' + max + '" aria-valuenow="' + hiW + '" tabindex="0"><span></span><span></span></div>'
      + '</div></div><div class="wk-bar-ticks">' + ticks + '</div></div>';
  };

  global._wkBarInit = function (id, onChange) {
    var root = document.getElementById(id);
    if (!root || root.dataset.wkInitialised === '1') return;
    root.dataset.wkInitialised = '1';
    var track = root.querySelector('.wk-bar-track'), sel = root.querySelector('.wk-bar-sel');
    var gripL = root.querySelector('.wk-bar-grip-l'), gripR = root.querySelector('.wk-bar-grip-r');
    var ticks = Array.from(root.querySelectorAll('.wk-tick'));
    if (!track || !sel || !gripL || !gripR) return;
    var min = Number(root.dataset.min), max = Number(root.dataset.max), n = max - min + 1;
    var ws = Number(root.dataset.ws), we = Number(root.dataset.we);
    var available = normaliseWeeks(min, max, (root.dataset.available || '').split(',').filter(Boolean));
    function nearest(w) {
      if (!available.length) return Math.max(min, Math.min(max, w));
      return available.reduce(function (best, x) { return Math.abs(x - w) < Math.abs(best - w) ? x : best; }, available[0]);
    }
    function fromX(x) { var r=track.getBoundingClientRect(); return min+Math.floor(Math.max(0,Math.min(1-1e-9,(x-r.left)/r.width))*n); }
    function paint() {
      sel.style.left=((ws-min)/n*100).toFixed(2)+'%'; sel.style.right=((max-we)/n*100).toFixed(2)+'%';
      ticks.forEach(function(t){var w=Number(t.dataset.week);t.classList.toggle('wk-tick-in',w>=ws&&w<=we);});
      gripL.setAttribute('aria-valuenow',String(ws)); gripR.setAttribute('aria-valuenow',String(we));
    }
    function emit(oldWs, oldWe) { if (ws !== oldWs || we !== oldWe) onChange(ws,we); }
    function drag(e, mode) {
      e.preventDefault(); var sx=e.touches?e.touches[0].clientX:e.clientX, oldWs=ws, oldWe=we, sw=ws, ew=we, anchor=fromX(sx);
      function move(ev){if(ev.cancelable)ev.preventDefault();var x=ev.touches?ev.touches[0].clientX:ev.clientX,w=nearest(fromX(x));
        if(mode==='lo')ws=Math.min(we,w); else if(mode==='hi')we=Math.max(ws,w); else {var d=w-nearest(anchor),span=ew-sw;ws=Math.max(min,Math.min(max-span,sw+d));we=ws+span;ws=nearest(ws);we=nearest(we);if(we<ws)we=ws;} paint();}
      function up(){document.removeEventListener('mousemove',move);document.removeEventListener('mouseup',up);document.removeEventListener('touchmove',move);document.removeEventListener('touchend',up);emit(oldWs,oldWe);}
      document.addEventListener('mousemove',move);document.addEventListener('mouseup',up);document.addEventListener('touchmove',move,{passive:false});document.addEventListener('touchend',up);
    }
    [['mousedown','lo',gripL],['mousedown','hi',gripR],['touchstart','lo',gripL],['touchstart','hi',gripR]].forEach(function(x){x[2].addEventListener(x[0],function(e){e.stopPropagation();drag(e,x[1]);},x[0]==='touchstart'?{passive:false}:undefined);});
    function key(e,mode){var d=(e.key==='ArrowRight'||e.key==='ArrowUp')?1:(e.key==='ArrowLeft'||e.key==='ArrowDown')?-1:0;if(!d)return;e.preventDefault();var ow=ws,oe=we;if(mode==='lo')ws=nearest(Math.max(min,Math.min(we,ws+d)));else we=nearest(Math.max(ws,Math.min(max,we+d)));paint();emit(ow,oe);}
    gripL.addEventListener('keydown',function(e){key(e,'lo');});gripR.addEventListener('keydown',function(e){key(e,'hi');});
    sel.addEventListener('mousedown',function(e){if(!e.target.closest('.wk-bar-grip'))drag(e,'move');});sel.addEventListener('touchstart',function(e){if(!e.target.closest('.wk-bar-grip'))drag(e,'move');},{passive:false});
    track.addEventListener('click',function(e){if(e.target.closest('.wk-bar-sel'))return;var ow=ws,oe=we,w=nearest(fromX(e.clientX));ws=w;we=w;paint();onChange(ws,we);});
    paint();
  };
})(window);
