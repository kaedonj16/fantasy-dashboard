/* Setup wizard: Phase 1 of the draft room visual redesign.
 *
 * Visual layer only. Every control writes through to the original setup inputs
 * (same ids draft_room.js has always read: drType, drSf, drOrder, drPpr, drTep,
 * drPassTd, drTeams, drRounds, drSlot, drCpuAdpSource, drKeeperSource,
 * drKeeperCount), dispatching 'change' so the existing listeners fire with
 * identical semantics. The roster mini-steppers forward clicks to the original
 * .dr-step-btn buttons draft_room.js renders into #drRosterSection; the capital
 * list reads the #drCapitalSection DOM draft_room.js renders and mutates it
 * through the same data-rm / data-add / data-addround controls. Nothing here
 * changes behavior, formulas, or grading.
 */
(function(){
  'use strict';
  var root = document.getElementById('wzStep1');
  if (!root) return;  // wizard markup absent
  if (root._wzInit) return;  // already wired on this DOM; fresh DOM (soft nav) re-inits
  root._wzInit = true;
  var $ = function(id){ return document.getElementById(id); };

  function esc(s){
    return String(s).replace(/[&<>"']/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }
  function ordinal(n){
    var s = ['th','st','nd','rd'], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }
  function fireChange(el){ if (el) el.dispatchEvent(new Event('change')); }
  // Set an original input's value and fire change (only when it actually changed).
  function setOrig(id, val){
    var el = $(id);
    if (!el || el.value === val) return false;
    el.value = val;
    fireChange(el);
    return true;
  }

  /* ── Step navigation ── */
  var stepCards = [$('wzStep1'), $('wzStep2'), $('wzStep3')];
  var stepDots = [$('wzStepDot1'), $('wzStepDot2'), $('wzStepDot3')];
  function goStep(n){
    stepCards.forEach(function(card, i){
      var k = i + 1;
      if (card) card.hidden = (k !== n);
      var dot = stepDots[i];
      if (!dot) return;
      dot.classList.toggle('on', k === n);
      dot.classList.toggle('done', k < n);
      var num = dot.querySelector('.n');
      if (num) num.textContent = k < n ? '\u2713' : String(k);
    });
    var cur = stepCards[n - 1];
    if (cur && cur.scrollIntoView) cur.scrollIntoView({block: 'start'});
  }
  $('wzToStep2').addEventListener('click', function(){ goStep(2); });
  $('wzToStep3').addEventListener('click', function(){ goStep(3); });
  $('wzBackToStep1').addEventListener('click', function(){ goStep(1); });
  $('wzBackToStep2').addEventListener('click', function(){ goStep(2); });

  /* ── Segmented controls: buttons carry data-val matching the original
         select's option values ── */
  var SEG_ORIG = {
    dtype: 'drType', qb: 'drSf', order: 'drOrder', ppr: 'drPpr',
    tep: 'drTep', ptd: 'drPassTd', ksrc: 'drKeeperSource'
  };
  function syncSeg(group){
    var seg = document.querySelector('[data-wz-seg="' + group + '"]');
    var el = $(SEG_ORIG[group]);
    if (!seg || !el) return;
    var v = el.value;
    Array.prototype.forEach.call(seg.querySelectorAll('button'), function(b){
      b.classList.toggle('on', b.getAttribute('data-val') === v);
    });
  }
  function syncAllSegs(){ Object.keys(SEG_ORIG).forEach(syncSeg); }
  function updateKeeperBox(){
    var box = $('wzKeeperBox');
    if (box) box.hidden = (($('drType') || {}).value !== 'keeper');
  }
  Object.keys(SEG_ORIG).forEach(function(group){
    var seg = document.querySelector('[data-wz-seg="' + group + '"]');
    if (!seg) return;
    seg.addEventListener('click', function(e){
      var b = e.target.closest('button');
      if (!b || b.classList.contains('on')) return;
      setOrig(SEG_ORIG[group], b.getAttribute('data-val'));
      syncSeg(group);
      if (group === 'dtype') updateKeeperBox();
      if (group === 'order') buildSlots();   // pick-order changes the slot note
      if (group === 'qb') syncRoster();      // SF toggle reseeds the roster
      markCustom();
    });
  });

  /* ── Steppers: teams / rounds / keepers write to the original inputs ── */
  var STEPPER_ORIG = { keepers: 'drKeeperCount', teams: 'drTeams', rounds: 'drRounds' };
  function syncStepper(st){
    var el = $(STEPPER_ORIG[st.getAttribute('data-wz-stepper')]);
    var vEl = st.querySelector('.val');
    if (el && vEl){
      var v = parseInt(el.value, 10);
      if (!isNaN(v)) vEl.textContent = String(v);
    }
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-wz-stepper]'), function(st){
    var name = st.getAttribute('data-wz-stepper');
    var origId = STEPPER_ORIG[name];
    var min = parseInt(st.getAttribute('data-min'), 10);
    var max = parseInt(st.getAttribute('data-max'), 10);
    st._wzSync = function(){ syncStepper(st); };
    st.addEventListener('click', function(e){
      var b = e.target.closest('button');
      if (!b) return;
      var el = $(origId);
      var cur = el ? parseInt(el.value, 10) : NaN;
      if (isNaN(cur)) cur = parseInt(st.querySelector('.val').textContent, 10) || min;
      var n = Math.max(min, Math.min(max, cur + parseInt(b.getAttribute('data-dir'), 10)));
      if (el){
        el.value = String(n);
        el.dispatchEvent(new Event('input'));  // drKeeperCount tracks dataset.touched on input
        fireChange(el);
      }
      syncStepper(st);   // read back: handlers may clamp or derive (rounds <-> bench)
      markCustom();
    });
  });
  function syncAllSteppers(){
    Array.prototype.forEach.call(document.querySelectorAll('[data-wz-stepper]'), function(st){
      if (st._wzSync) st._wzSync();
    });
  }

  /* ── Slot picker: visual boxes 1..N (+ Random), writes to #drSlot ── */
  var slotPicker = $('wzSlotPicker'), slotNote = $('wzSlotNote');
  function orderVal(){ return ($('drOrder') || {}).value || 'snake'; }
  function buildSlots(){
    var sel = $('drSlot');
    if (!sel || !slotPicker || !slotNote) return;
    var teams = 0, nums = [];
    Array.prototype.forEach.call(sel.options, function(o){
      if (o.value === 'random') return;
      var n = parseInt(o.value, 10);
      if (n >= 1){ nums.push(n); if (n > teams) teams = n; }
    });
    nums.sort(function(a, b){ return a - b; });
    var cur = sel.value;
    slotPicker.innerHTML = '';
    nums.forEach(function(i){
      var d = document.createElement('button');
      d.type = 'button';
      d.className = 'wz-slotbox' + (String(i) === cur ? ' sel' : '');
      d.setAttribute('role', 'radio');
      d.setAttribute('aria-checked', String(i) === cur ? 'true' : 'false');
      d.setAttribute('data-slot', String(i));
      d.innerHTML = i + '<small>' + (i === 1 ? '1ST' : (i === teams ? 'TURN' : 'PK')) + '</small>';
      slotPicker.appendChild(d);
    });
    var r = document.createElement('button');
    r.type = 'button';
    r.className = 'wz-slotbox random' + (cur === 'random' ? ' sel' : '');
    r.setAttribute('role', 'radio');
    r.setAttribute('aria-checked', cur === 'random' ? 'true' : 'false');
    r.setAttribute('data-slot', 'random');
    r.setAttribute('title', 'Draw a random draft slot when the draft starts');
    r.textContent = 'Random';
    slotPicker.appendChild(r);
    var msg, s = parseInt(cur, 10), order = orderVal();
    if (cur === 'random' || isNaN(s)){
      msg = 'Random: your seat is drawn when the draft starts, and your picks follow from it.';
    } else if (order === 'snake'){
      var turn = teams * 2 - s + 1;
      msg = 'Pick <b>' + s + '</b>: you draft ' + ordinal(s) + ', then ' + ordinal(turn) + ' on the turn (snake).';
    } else if (order === 'linear'){
      msg = 'Pick <b>' + s + '</b>: same slot every round (linear). Round 2 pick is ' + ordinal(teams + s) + ' overall.';
    } else {
      msg = 'Pick <b>' + s + '</b>: snake, except round 3 reverses back to ' + ordinal(s) + ' (3rd round reversal).';
    }
    slotNote.innerHTML = msg;
  }
  if (slotPicker){
    slotPicker.addEventListener('click', function(e){
      var b = e.target.closest('[data-slot]');
      if (!b) return;
      setOrig('drSlot', b.getAttribute('data-slot'));
      buildSlots();
      markCustom();
      // The original drSlot change listener re-renders the capital section;
      // the observer below rebuilds the wizard capital list.
    });
  }

  /* ── Roster grid: mini-steppers forward clicks to the original
         .dr-step-btn buttons draft_room.js renders into #drRosterSection ── */
  var CORE_ORDER = ['QB', 'SF', 'RB', 'WR', 'TE', 'FLEX', 'K', 'DEF', 'BN'];
  var CORE_LABEL = {QB: 'QB', SF: 'SF', RB: 'RB', WR: 'WR', TE: 'TE', FLEX: 'FLEX', K: 'K', DEF: 'DEF', BN: 'BN'};
  var ROSTER_LABEL_KEY = {QB: 'QB', Superflex: 'SF', RB: 'RB', WR: 'WR', TE: 'TE', FLEX: 'FLEX',
    K: 'K', DEF: 'DEF', Bench: 'BN', IR: 'IR', Taxi: 'TAXI', IDP: 'IDP'};
  function readOrigRoster(){
    var sec = $('drRosterSection');
    var out = {rows: [], locked: true, tag: ''};
    if (!sec) return out;
    var tagEl = sec.querySelector('.dr-roster-src-tag');
    if (tagEl) out.tag = tagEl.textContent.trim();
    out.locked = !sec.querySelector('.dr-step-btn');
    Array.prototype.forEach.call(sec.querySelectorAll('.dr-srow'), function(row){
      var b = row.querySelector('.dr-step-btn');
      var v = row.querySelector('.dr-step-val, .dr-step-val-ro');
      var labEl = row.querySelector('.dr-srow-label');
      out.rows.push({
        key: b ? b.getAttribute('data-key') : null,
        label: labEl ? labEl.textContent.trim() : '',
        val: v ? (parseInt(v.textContent, 10) || 0) : 0
      });
    });
    return out;
  }
  function rosterKeyOf(r){ return r.key || ROSTER_LABEL_KEY[r.label] || null; }
  function rosterVal(k){
    var rows = readOrigRoster().rows;
    for (var i = 0; i < rows.length; i++){
      if (rosterKeyOf(rows[i]) === k) return rows[i].val;
    }
    return 0;
  }
  function syncRoster(){
    var grid = $('wzRosterGrid');
    if (!grid) return;
    var info = readOrigRoster();
    var byKey = {};
    info.rows.forEach(function(r){
      var k = rosterKeyOf(r);
      if (k) byKey[k] = r.val;
    });
    var order = CORE_ORDER.concat(Object.keys(byKey).filter(function(k){ return CORE_ORDER.indexOf(k) < 0; }));
    grid.innerHTML = '';
    order.forEach(function(k){
      if (byKey[k] == null) return;
      var lab = CORE_LABEL[k] || k;
      var cell = document.createElement('div');
      cell.className = 'wz-rslot';
      cell.innerHTML = '<span class="wz-rl">' + esc(lab) + '</span>' +
        '<span class="wz-mini-stepper" data-rk="' + esc(k) + '">' +
        '<button type="button" data-dir="-1" aria-label="Fewer ' + esc(lab) + '"' + (info.locked ? ' disabled' : '') + '>&minus;</button>' +
        '<span class="val">' + byKey[k] + '</span>' +
        '<button type="button" data-dir="1" aria-label="More ' + esc(lab) + '"' + (info.locked ? ' disabled' : '') + '>+</button></span>';
      grid.appendChild(cell);
    });
    var lock = $('wzRosterLock');
    if (lock){
      if (info.locked){
        var cust = $('drRosterCustomize');
        lock.hidden = false;
        lock.innerHTML = '<span>' + (info.tag ? esc(info.tag) + ' roster' : 'Roster comes from your league settings') + '.</span>' +
          (cust ? '<button type="button" class="wz-linkbtn" id="wzRosterCustomBtn">Customize</button>' : '');
        var cb = $('wzRosterCustomBtn');
        if (cb && cust) cb.addEventListener('click', function(){ cust.click(); });
      } else {
        var rst = $('drRosterReset');
        if (rst && info.tag){
          lock.hidden = false;
          lock.innerHTML = '<span>' + esc(info.tag) + '</span>' +
            '<button type="button" class="wz-linkbtn" id="wzRosterResetBtn">Reset to league</button>';
          $('wzRosterResetBtn').addEventListener('click', function(){ rst.click(); });
        } else {
          lock.hidden = true;
          lock.innerHTML = '';
        }
      }
    }
  }
  var rosterGrid = $('wzRosterGrid');
  if (rosterGrid){
    rosterGrid.addEventListener('click', function(e){
      var b = e.target.closest('button');
      if (!b || b.disabled) return;
      var rk = b.parentNode.getAttribute('data-rk');
      var d = b.getAttribute('data-dir');
      var origBtn = document.querySelector('#drRosterSection .dr-step-btn[data-key="' + rk + '"][data-d="' + d + '"]');
      if (!origBtn) return;
      origBtn.click();   // draft_room.js updates its roster state and re-renders, synchronously
      syncSeg('qb');     // SF edits retarget the drSf select directly
      syncRoster();
      rebuildCapital();  // slot changes re-derive rounds, which renumbers picks
      markCustom();
    });
  }

  /* ── Draft capital: wizard list reads the #drCapitalSection DOM that
         draft_room.js renders, and mutates it through the same controls ── */
  var wzRemoved = [];     // [{pn, round}] picks marked traded-away, newest last
  var wzAddRound = null;  // round whose inline add-grid is open in the wizard
  var wzCapSig = null;
  function capSig(){
    return [($('drTeams') || {}).value, ($('drRounds') || {}).value,
            ($('drOrder') || {}).value, ($('drSlot') || {}).value].join('|');
  }
  function hiddenPickerRound(){
    var b = document.querySelector('#drCapitalSection .dr-cap-row.is-open .dr-cap-addbtn');
    return b ? parseInt(b.getAttribute('data-addround'), 10) : null;
  }
  function ensureLateOpen(){
    var sec = $('drCapitalSection');
    if (!sec) return;
    var late = sec.querySelector('.dr-cap-late');
    if (late && !late.classList.contains('is-open')){
      var t = $('drCapLateToggle');
      if (t) t.click();
    }
  }
  function readCapital(){
    var sec = $('drCapitalSection');
    var out = {rounds: 0, random: false, rows: [], owned: 0, extra: 0};
    if (!sec) return out;
    if ((($('drSlot') || {}).value) === 'random' || !sec.querySelector('.dr-cap-row')){
      out.random = true;
      return out;
    }
    ensureLateOpen();
    out.rounds = parseInt(($('drRounds') || {}).value, 10) || 0;
    Array.prototype.forEach.call(sec.querySelectorAll('.dr-cap-row'), function(row){
      var lbl = row.querySelector('.dr-cap-rlabel');
      var m = lbl && lbl.textContent.match(/R(\d+)/);
      if (!m) return;
      var picks = [];
      Array.prototype.forEach.call(row.querySelectorAll('.dr-cap-pill'), function(pill){
        var pn = parseInt(pill.getAttribute('data-rm'), 10);
        if (!pn) return;
        var traded = pill.classList.contains('dr-cap-pill-traded');
        picks.push({pn: pn, traded: traded});
        out.owned++;
        if (traded) out.extra++;
      });
      out.rows.push({round: parseInt(m[1], 10), picks: picks});
    });
    return out;
  }
  function fmtPick(r, pn, teams){
    return r + '.' + String(pn - (r - 1) * teams).padStart(2, '0');
  }
  function orderLabel(o){
    return o === 'linear' ? 'linear' : (o === '3rr' ? '3rd round reversal' : 'snake');
  }
  function addGridCells(){
    var cells = '';
    Array.prototype.forEach.call(
      document.querySelectorAll('#drCapitalSection .dr-cap-picker .dr-cap-slot'), function(b){
        var pn = b.getAttribute('data-add');
        var n = b.textContent.trim();
        cells += '<button type="button" class="wz-slotbox' +
          (b.classList.contains('on') ? ' on' : '') +
          (b.classList.contains('home') ? ' home' : '') +
          '" data-wz-slotpick="' + pn + '" title="Pick ' + pn + ' overall">' + esc(n) + '</button>';
      });
    return cells || '<span class="wz-fhint">No slots</span>';
  }
  function openHiddenPicker(r){
    var cur = hiddenPickerRound();
    if (cur === r) return;
    var sec = $('drCapitalSection');
    if (!sec) return;
    if (cur != null){
      var cb = sec.querySelector('.dr-cap-addbtn[data-addround="' + cur + '"]');
      if (cb) cb.click();
    }
    var tb = sec.querySelector('.dr-cap-addbtn[data-addround="' + r + '"]');
    if (tb) tb.click();
  }
  function closeHiddenPicker(){
    var cur = hiddenPickerRound();
    if (cur == null) return;
    var cb = document.querySelector('#drCapitalSection .dr-cap-addbtn[data-addround="' + cur + '"]');
    if (cb) cb.click();
  }
  function wzAddPick(r, pn){
    openHiddenPicker(r);
    var b = document.querySelector('#drCapitalSection .dr-cap-slot[data-add="' + pn + '"]:not(.on)');
    if (b) b.click();
    closeHiddenPicker();
  }
  function rebuildCapital(){
    var list = $('wzPickList'), summary = $('wzCapSummary');
    if (!list || !summary) return;
    var sig = capSig();
    if (sig !== wzCapSig){ wzCapSig = sig; wzRemoved = []; wzAddRound = null; }
    var cap = readCapital();
    var slot = parseInt(($('drSlot') || {}).value, 10) || 0;
    var teams = parseInt(($('drTeams') || {}).value, 10) || 0;
    var order = ($('drOrder') || {}).value || 'snake';
    if (cap.random){
      summary.textContent = 'Your pick is random. You will be assigned a seat when the draft starts, and your draft capital will be set from it.';
      list.innerHTML = '';
      return;
    }
    var removedByRound = {};
    wzRemoved.forEach(function(rem){
      (removedByRound[rem.round] = removedByRound[rem.round] || []).push(rem);
    });
    var html = '';
    for (var r = 1; r <= cap.rounds; r++){
      var rowPicks = [];
      cap.rows.forEach(function(row){ if (row.round === r) rowPicks = row.picks; });
      var natural = null, extras = [];
      rowPicks.forEach(function(p){
        if (p.traded) extras.push(p);
        else if (!natural) natural = p;
        else extras.push(p);
      });
      var removed = removedByRound[r] || [];
      var cells = '';
      if (natural && !removed.length){
        cells += '<span class="wz-pbadge">' + fmtPick(r, natural.pn, teams) +
          ' <span class="wz-ov">Pick ' + natural.pn + ' overall</span></span>';
      }
      removed.forEach(function(rem){
        cells += '<span class="wz-pbadge traded"><span class="wz-pk">' + fmtPick(rem.round, rem.pn, teams) +
          '</span> <span class="wz-ov">was Pick ' + rem.pn + '</span> <span class="wz-tag">traded away</span></span>';
      });
      extras.forEach(function(p){
        cells += '<span class="wz-pbadge extra">' + fmtPick(r, p.pn, teams) +
          ' <span class="wz-ov">Pick ' + p.pn + ' overall</span> <span class="wz-tag">via trade</span></span>';
      });
      if (!cells) cells = '<span class="wz-fhint" style="margin:0">No picks</span>';
      var acts = '';
      extras.forEach(function(p){
        acts += '<button type="button" class="wz-mini-btn" data-wz-rx="' + p.pn + '" aria-label="Remove added pick">x</button>';
      });
      if (removed.length){
        acts += '<button type="button" class="wz-mini-btn undo" data-wz-undo="' + r + '">Undo</button>';
      } else if (natural){
        acts += '<button type="button" class="wz-mini-btn" data-wz-rm="' + r + '" data-pn="' + natural.pn + '" aria-label="Remove pick">x remove</button>';
      }
      acts += '<button type="button" class="wz-mini-btn" data-wz-add="' + r + '">+ add pick</button>';
      html += '<div class="wz-prow"><span class="wz-rd">Round ' + r + '</span><span class="wz-picks">' + cells +
        '</span><span class="wz-acts">' + acts + '</span></div>';
      if (wzAddRound === r){
        html += '<div class="wz-addgrid">' + addGridCells() + '</div>';
      }
    }
    list.innerHTML = html;
    var away = wzRemoved.length, added = cap.extra;
    var bits = [];
    if (away) bits.push(away + ' traded away');
    if (added) bits.push(added + ' added via trade');
    var tail = bits.length ? ' (' + bits.join(', ') + ')' : ', one per round';
    summary.innerHTML = 'You pick <b>' + ordinal(slot) + '</b> (' + orderLabel(order) + '). ' +
      'You have <b>' + cap.owned + ' pick' + (cap.owned === 1 ? '' : 's') + '</b> across ' +
      cap.rounds + ' rounds' + tail + '.';
  }
  var pickList = $('wzPickList');
  if (pickList){
    pickList.addEventListener('click', function(e){
      var b = e.target.closest('button');
      if (!b) return;
      var sec = $('drCapitalSection');
      if (b.hasAttribute('data-wz-rm')){
        var pn = b.getAttribute('data-pn');
        var r = parseInt(b.getAttribute('data-wz-rm'), 10);
        wzRemoved.push({pn: parseInt(pn, 10), round: r});
        var pill = sec && sec.querySelector('.dr-cap-pill[data-rm="' + pn + '"]');
        if (pill) pill.click();
        rebuildCapital();
        markCustom();
        return;
      }
      if (b.hasAttribute('data-wz-undo')){
        var ru = parseInt(b.getAttribute('data-wz-undo'), 10);
        for (var i = wzRemoved.length - 1; i >= 0; i--){
          if (wzRemoved[i].round === ru){
            var rem = wzRemoved.splice(i, 1)[0];
            wzAddPick(rem.round, rem.pn);
            break;
          }
        }
        rebuildCapital();
        markCustom();
        return;
      }
      if (b.hasAttribute('data-wz-rx')){
        var px = b.getAttribute('data-wz-rx');
        var pillx = sec && sec.querySelector('.dr-cap-pill[data-rm="' + px + '"]');
        if (pillx) pillx.click();
        rebuildCapital();
        markCustom();
        return;
      }
      if (b.hasAttribute('data-wz-add')){
        var ra = parseInt(b.getAttribute('data-wz-add'), 10);
        if (wzAddRound === ra){ wzAddRound = null; closeHiddenPicker(); }
        else { wzAddRound = ra; openHiddenPicker(ra); }
        rebuildCapital();
        markCustom();
        return;
      }
      if (b.hasAttribute('data-wz-slotpick')){
        var ps = b.getAttribute('data-wz-slotpick');
        var slotBtn = sec && sec.querySelector('.dr-cap-slot[data-add="' + ps + '"]');
        if (slotBtn) slotBtn.click();
        rebuildCapital();
        markCustom();
      }
    });
  }

  /* ── Presets ── */
  var WZ_PRESETS = {
    ppr10:   {dtype: 'redraft', sf: '0', order: 'snake', ppr: '1', tep: '0', ptd: '4',
              teams: '10', rounds: '15', slot: '9', cpu: 'consensus',
              roster: {QB: 1, SF: 0, RB: 2, WR: 2, TE: 1, FLEX: 2, K: 1, DEF: 1, BN: 5}},
    sf12:    {dtype: 'redraft', sf: '1', order: 'snake', ppr: '1', tep: '1', ptd: '4',
              teams: '12', rounds: '15', slot: '6', cpu: 'consensus',
              roster: {QB: 1, SF: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DEF: 1, BN: 5}},
    dynasty: {dtype: 'startup', sf: '1', order: 'snake', ppr: '1', tep: '0', ptd: '4',
              teams: '12', rounds: '25', slot: '4', cpu: 'sleeper',
              roster: {QB: 1, SF: 1, RB: 2, WR: 3, TE: 1, FLEX: 2, K: 0, DEF: 0, BN: 15}},
    keeper2: {dtype: 'keeper', ksrc: 'assistant', keepers: '2', sf: '0', order: 'snake', ppr: '0.5', tep: '0', ptd: '4',
              teams: '10', rounds: '15', slot: '7', cpu: 'espn',
              roster: {QB: 1, SF: 0, RB: 2, WR: 2, TE: 1, FLEX: 2, K: 1, DEF: 1, BN: 5}}
  };
  var activePreset = null;
  function presetCards(){ return document.querySelectorAll('.wz-pcard'); }
  function selectPresetUI(key){
    activePreset = key;
    Array.prototype.forEach.call(presetCards(), function(c){
      var k = c.getAttribute('data-preset');
      c.classList.toggle('on', k === key);
      if (k === 'custom') c.hidden = (key !== 'custom');
    });
    var label = $('wzPresetCurName');
    var nameEl = document.querySelector('.wz-pcard[data-preset="' + key + '"] .wz-pname');
    if (label) label.textContent = nameEl ? nameEl.textContent.trim() : 'Custom';
  }
  function updateCustomPills(){
    var cp = $('wzCustomPills');
    if (cp){
      var qb = (($('drSf') || {}).value === '1') ? 'SF' : '1QB';
      cp.textContent = (($('drTeams') || {}).value || '?') + ' teams \u00B7 ' + qb +
        ' \u00B7 ' + (($('drRounds') || {}).value || '?') + ' rds';
    }
  }
  function markCustom(){
    if (activePreset !== 'custom') selectPresetUI('custom');
    updateCustomPills();
  }
  function applyRosterMap(map){
    // A preset defines the whole format: unlock a league-seeded roster first,
    // exactly like the original in-app roster presets do.
    var cust = $('drRosterCustomize');
    if (cust && !$('drRosterSection').querySelector('.dr-step-btn')) cust.click();
    Object.keys(map).forEach(function(k){
      var want = map[k];
      for (var guard = 0; guard < 40; guard++){
        var cur = rosterVal(k);
        if (cur === want) break;
        var d = want > cur ? '1' : '-1';
        var btn = document.querySelector('#drRosterSection .dr-step-btn[data-key="' + k + '"][data-d="' + d + '"]');
        if (!btn) break;
        btn.click();
      }
    });
  }
  function applyPreset(key){
    var pr = WZ_PRESETS[key];
    if (!pr) return;
    setPresetOpen(false);
    setOrig('drType', pr.dtype);
    if (pr.dtype === 'keeper'){
      setOrig('drKeeperSource', pr.ksrc || 'assistant');
      var kc = $('drKeeperCount');
      if (kc){
        kc.value = pr.keepers;
        kc.dispatchEvent(new Event('input'));
        fireChange(kc);
      }
    }
    setOrig('drSf', pr.sf);
    setOrig('drOrder', pr.order);
    setOrig('drPpr', pr.ppr);
    setOrig('drTep', pr.tep);
    setOrig('drPassTd', pr.ptd);
    setOrig('drTeams', pr.teams);
    setOrig('drSlot', pr.slot);
    var cpu = $('drCpuAdpSource');
    if (cpu && cpu.querySelector('option[value="' + pr.cpu + '"]')) setOrig('drCpuAdpSource', pr.cpu);
    applyRosterMap(pr.roster);
    setOrig('drRounds', pr.rounds);   // last: rounds<->bench two-way sync derives BN
    fullSync();
    selectPresetUI(key);
    if (!$('wzStep1').hidden) goStep(2);
  }
  function currentMatches(pr){
    function v(id){ return (($(id) || {}).value); }
    if (v('drType') !== pr.dtype || v('drSf') !== pr.sf || v('drOrder') !== pr.order ||
        v('drPpr') !== pr.ppr || v('drTep') !== pr.tep || v('drPassTd') !== pr.ptd ||
        v('drTeams') !== pr.teams || v('drRounds') !== pr.rounds || v('drSlot') !== pr.slot ||
        v('drCpuAdpSource') !== pr.cpu) return false;
    if (pr.dtype === 'keeper' &&
        (v('drKeeperSource') !== (pr.ksrc || 'assistant') || v('drKeeperCount') !== String(pr.keepers))) return false;
    var rm = pr.roster;
    for (var k in rm){
      if (rm.hasOwnProperty(k) && rosterVal(k) !== rm[k]) return false;
    }
    return true;
  }
  function detectPreset(){
    for (var key in WZ_PRESETS){
      if (WZ_PRESETS.hasOwnProperty(key) && currentMatches(WZ_PRESETS[key])){
        selectPresetUI(key);
        return;
      }
    }
    markCustom();
  }
  var presetWrap = $('wzPresetWrap'), presetHead = $('wzPresetHead'), presetPanel = $('wzPresetPanel');
  function setPresetOpen(open){
    if (!presetPanel || !presetWrap || !presetHead) return;
    presetPanel.classList.toggle('open', open);
    presetWrap.classList.toggle('open', open);
    presetHead.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  if (presetHead){
    presetHead.addEventListener('click', function(){
      setPresetOpen(!presetPanel.classList.contains('open'));
    });
  }
  Array.prototype.forEach.call(presetCards(), function(p){
    p.addEventListener('click', function(){
      var key = p.getAttribute('data-preset');
      if (key === 'custom'){ selectPresetUI('custom'); return; }
      applyPreset(key);
    });
  });
  var cpuSel = $('drCpuAdpSource');
  if (cpuSel) cpuSel.addEventListener('change', markCustom);

  /* ── Live connect card: one button for the connected league, driving the
         exact same flow as the original #drConnect button ── */
  function initLive(){
    var card = $('wzLiveCard');
    if (!card) return;
    var cfg = window.__draftCfg || {};
    if (cfg.isGuest || !cfg.leagueId){ card.hidden = true; return; }
    var title = $('wzLiveTitle');
    var name = cfg.leagueName || cfg.league_name || '';
    if (title && name) title.textContent = 'Sync ' + name + '\u2019s live draft';
    var go = $('wzLiveGo');
    var list = $('drLiveList');
    if (list && go){
      new MutationObserver(function(){ go.disabled = false; }).observe(list, {childList: true});
    }
    if (go){
      go.addEventListener('click', function(){
        go.disabled = true;
        var conn = $('drConnect');
        if (conn) conn.click();   // detectLive(): identical flow, results render into #drLiveList
        if (list) list.scrollIntoView({block: 'center', behavior: 'smooth'});
      });
    }
  }

  /* ── Resume: the page auto-resumes a session draft on load, so the link
         reloads into that draft when one exists ── */
  function initResume(){
    var wrap = $('wzResumeWrap');
    if (!wrap) return;
    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem('dr_' + location.pathname) || 'null'); } catch (e) {}
    if (saved && saved.teams && saved.picks){
      wrap.hidden = false;
      var link = $('wzResume');
      if (link) link.addEventListener('click', function(e){
        e.preventDefault();
        location.reload();
      });
    }
  }

  /* ── Full sync + observers: stay in lockstep with draft_room.js ── */
  function fullSync(){
    syncAllSegs();
    updateKeeperBox();
    syncAllSteppers();
    buildSlots();
    syncRoster();
    rebuildCapital();
    updateCustomPills();
  }
  function observe(){
    var rsec = $('drRosterSection');
    if (rsec) new MutationObserver(function(){ syncRoster(); }).observe(rsec, {childList: true, subtree: true});
    var csec = $('drCapitalSection');
    if (csec) new MutationObserver(function(){ rebuildCapital(); }).observe(csec, {childList: true});
    var ssel = $('drSlot');
    if (ssel) new MutationObserver(function(){ buildSlots(); }).observe(ssel, {childList: true});
    // showSetup / openEditSetup / closeEditSetup hydrate the original inputs
    // directly; re-sync the wizard chrome when the setup is shown or hidden.
    new MutationObserver(function(){ fullSync(); }).observe($('drSetup'), {attributes: true, attributeFilter: ['style', 'class']});
  }

  /* ── Init (runs after draft_room.js: deferred scripts execute in order) ── */
  // Keeper draft type may be removed entirely for non-keeper leagues.
  if (!document.querySelector('#drType option[value="keeper"]')){
    var kb = $('wzSegKeeper');
    if (kb) kb.style.display = 'none';
    var kcard = document.querySelector('.wz-pcard[data-preset="keeper2"]');
    if (kcard) kcard.style.display = 'none';
  }
  observe();
  initLive();
  initResume();
  fullSync();
  detectPreset();
  goStep(1);
})();
