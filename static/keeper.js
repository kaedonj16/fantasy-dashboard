/* Keeper Assistant - client interactivity.
 *
 * Mirrors utils/keeper_value.py so the keeper limit, cost rules and per-player
 * round edits recompute live without a server round-trip. Keep the math here in
 * sync with the Python engine (tests/test_keeper_value.py is the source of truth).
 */
(function () {
  "use strict";
  var seedEl = document.getElementById("kpr-seed");
  if (!seedEl) return;
  var seed;
  try { seed = JSON.parse(seedEl.textContent || "{}"); } catch (e) { return; }

  var players = (seed.players || []).map(function (p) { return Object.assign({}, p); });
  var leagueSize = seed.leagueSize || 12;
  var numRounds = seed.numRounds || 15;

  var $ = function (id) { return document.getElementById(id); };
  var elLim = $("kpr-lim"), elLimN = $("kpr-limn"), elTot = $("kpr-tot"),
      elList = $("kpr-list"), elTbody = $("kpr-tbody"),
      elCost = $("kpr-cost"), elEsc = $("kpr-esc"), elUndr = $("kpr-undr"),
      elOpr = $("kpr-opr"),
      elOpt = $("kpr-optimizer"), elTbl = $("kpr-table"),
      elViewOpt = $("kpr-view-opt"), elViewTbl = $("kpr-view-tbl");

  if (elLim) { elLim.max = String(Math.max(1, players.length)); elLim.value = String(Math.min(seed.maxKeepers || 2, players.length)); }

  var CHECK = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>';

  function rules() {
    var undr = parseInt(elUndr && elUndr.value, 10);
    return {
      roundOffset: parseInt(elCost && elCost.value, 10) || 0,
      escalation: parseInt(elEsc && elEsc.value, 10) || 0,
      undraftedRound: (undr > 0 ? undr : numRounds),
      onePerRound: elOpr ? !!elOpr.checked : (seed.onePerRound !== false),
      keepAt: 2, passAt: 0
    };
  }

  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

  function costRound(p, r) {
    var base = (p.draftedRound == null || p.draftedRound === "")
      ? r.undraftedRound
      : (parseInt(p.draftedRound, 10) + r.roundOffset);
    var cost = base - Math.max(0, p.yearsKept || 0) * r.escalation;
    return clamp(cost, 1, numRounds);
  }
  function marketRound(p) {
    if (!p.adpOverall || p.adpOverall <= 0) return null;
    return Math.ceil(p.adpOverall / leagueSize);
  }
  function verdict(s, r) {
    if (s == null) return "pass";
    if (s >= r.keepAt) return "keep";
    if (s < r.passAt) return "pass";
    return "toss";
  }
  // Mirror of utils.keeper_value.resolve_cost_collisions: give every kept row a
  // unique cost round (one pick per round), bumping duplicates to the nearest
  // open round — earlier (costlier) preferred — and re-pricing surplus/verdict.
  // Records what moved (for the heads-up note). Mutates the kept rows in place.
  var lastBumps = [];
  function resolveCollisions(rows, r) {
    lastBumps = [];
    var kept = rows.filter(function (row) { return row.keep; });
    kept.sort(function (a, b) {
      var sa = a.surplus == null ? -9999 : a.surplus, sb = b.surplus == null ? -9999 : b.surplus;
      return (sb - sa) || ((b.p.value || 0) - (a.p.value || 0));
    });
    var taken = {};
    kept.forEach(function (row) {
      var c = row.cost;
      if (c >= 1 && c <= numRounds && !taken[c]) { taken[c] = 1; return; }
      var placed = null;
      for (var d = 1; d < numRounds && placed == null; d++) {
        var earlier = c - d, later = c + d;
        if (earlier >= 1 && earlier <= numRounds && !taken[earlier]) placed = earlier;
        else if (later >= 1 && later <= numRounds && !taken[later]) placed = later;
      }
      if (placed == null) placed = c;   // degenerate: more keepers than rounds
      if (placed !== c) lastBumps.push({ name: row.p.name, from: c, to: placed });
      row.cost = placed;
      taken[placed] = 1;
      if (row.mkt != null) { row.surplus = row.cost - row.mkt; row.verdict = verdict(row.surplus, r); }
    });
  }
  function fmt(n) { return (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n) + " rd"; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; }); }

  function compute() {
    var r = rules();
    var rows = players.map(function (p) {
      var cost = costRound(p, r);
      var mkt = marketRound(p);
      var surplus = mkt == null ? null : (cost - mkt);
      return { p: p, cost: cost, mkt: mkt, surplus: surplus, verdict: verdict(surplus, r) };
    });
    rows.sort(function (a, b) {
      var sa = a.surplus == null ? -9999 : a.surplus, sb = b.surplus == null ? -9999 : b.surplus;
      return (sb - sa) || ((b.p.value || 0) - (a.p.value || 0));
    });
    var limit = parseInt(elLim && elLim.value, 10); if (isNaN(limit)) limit = 0;
    rows.forEach(function (row, i) { row.keep = i < limit && row.surplus != null && row.surplus > 0; });
    if (r.onePerRound) resolveCollisions(rows, r);
    else lastBumps = [];
    return rows;
  }

  function maxAbs(rows) {
    return rows.reduce(function (m, r) { return Math.max(m, Math.abs(r.surplus || 0)); }, 1);
  }

  function collisionWarning(rows) {
    // With one-per-round on, collisions are resolved (not just warned): show what
    // got bumped so the manager sees why a cost moved.
    if (rules().onePerRound) {
      if (!lastBumps.length) return "";
      var moved = lastBumps.map(function (b) {
        return esc(b.name) + " → R" + b.to + " (from R" + b.from + ")";
      });
      return '<div class="kpr-warn">One keeper per round: bumped ' + moved.join("; ") + ".</div>";
    }
    var byRound = {};
    rows.forEach(function (r) { if (r.keep) { (byRound[r.cost] = byRound[r.cost] || []).push(r); } });
    var clashes = Object.keys(byRound).filter(function (rd) { return byRound[rd].length > 1; });
    if (!clashes.length) return "";
    var parts = clashes.map(function (rd) { return byRound[rd].length + " keepers cost Round " + rd; });
    return '<div class="kpr-warn">Heads up: ' + parts.join("; ") +
      ". Many leagues bump a duplicate to the next open round.</div>";
  }

  function renderOptimizer(rows) {
    if (!rows.length) { elList.innerHTML = '<div class="kpr-empty">No players on this roster yet.</div>'; elTot.textContent = "+0 rd"; return; }
    var total = 0;
    var body = rows.map(function (row) {
      if (row.keep) total += row.surplus;
      var cls = row.keep ? "keep" : "cut";
      var sval = row.surplus == null ? "-" : fmt(row.surplus);
      var mkt = row.mkt == null ? "off-board" : ("market R" + row.mkt);
      return '<div class="kpr-row ' + cls + '">' +
        '<div class="kpr-chk">' + (row.keep ? CHECK : "") + "</div>" +
        '<div><div class="kpr-nm">' + esc(row.p.name) + '</div>' +
        '<div class="kpr-sub">' + esc(row.p.pos || "") + " · cost R" + row.cost + " · " + mkt + "</div></div>" +
        '<div class="kpr-mid">' + (row.keep ? "Keeping" : "Back in draft") + "</div>" +
        '<div class="kpr-val ' + (row.keep ? "keep" : "pass") + '">' + sval + "</div>" +
        "</div>";
    }).join("");
    elList.innerHTML = collisionWarning(rows) + body;
    elTot.textContent = fmt(total);
    if (elLimN) elLimN.textContent = elLim ? elLim.value : "0";
  }

  // Derived-cell builders, shared by the full render and the in-place patch so
  // an inline edit updates the numbers without rebuilding (and thus locking or
  // blurring) the round / years-kept inputs the manager is typing in.
  function mktCell(row) {
    return row.mkt == null ? '<span style="color:var(--text-muted)">-</span>'
      : ("Round " + row.mkt + ' <span style="color:var(--text-muted)">· ' + Math.round(row.p.value || 0) + "</span>");
  }
  function surpCell(row, mx) {
    var w = Math.round(Math.abs(row.surplus || 0) / mx * 100);
    var sColor = "var(--text-muted)";
    if (row.verdict === "keep") sColor = "var(--win,#15803d)";
    else if (row.verdict === "toss") sColor = "var(--inj-q,#ca8a04)";
    var sval = row.surplus == null ? "-" : fmt(row.surplus);
    return '<span class="kpr-bar"><i style="width:' + w + "%;background:" + sColor + '"></i></span>' +
      '<b style="color:' + sColor + '">' + sval + "</b>";
  }
  function verdictCell(row) {
    var vlabel = { keep: "KEEP", toss: "TOSS-UP", pass: "PASS" }[row.verdict];
    return '<span class="kpr-verdict ' + row.verdict + '"><span class="d"></span>' + vlabel + "</span>";
  }

  function renderTable(rows) {
    var mx = maxAbs(rows);
    elTbody.innerHTML = rows.map(function (row) {
      var pos = (row.p.pos || "").toUpperCase();
      var did = (row.p.draftedRound == null || row.p.draftedRound === "") ? "" : String(row.p.draftedRound);
      // Always an editable input (even for an auto-detected round) so a wrong
      // value can always be corrected — it never locks into static text.
      var draftedTxt = '<input class="kpr-drnd" type="number" min="1" data-id="' + esc(row.p.id) +
        '" placeholder="R?" value="' + esc(did) + '" aria-label="Drafted round">';
      draftedTxt += '<span class="kpr-dot">·</span>kept <input class="kpr-yrs" type="number" min="0" max="15" data-id="' +
        esc(row.p.id) + '" value="' + (row.p.yearsKept || 0) + '" aria-label="Years kept"> yr';
      return '<tr data-pid="' + esc(row.p.id) + '">' +
        '<td><div class="kpr-nm-line"><span class="kpr-pos ' + esc(pos) + '">' + (esc(pos) || "-") + "</span>" +
        '<span class="kpr-nm">' + esc(row.p.name) + '</span></div><div class="kpr-sub">' + draftedTxt + "</div></td>" +
        '<td class="r kpr-c-cost">Round ' + row.cost + "</td>" +
        '<td class="r kpr-c-mkt">' + mktCell(row) + "</td>" +
        '<td class="r kpr-c-surp">' + surpCell(row, mx) + "</td>" +
        '<td class="r kpr-c-verd">' + verdictCell(row) + "</td>" +
        "</tr>";
    }).join("");
    bindInlineInput(".kpr-drnd", function (pl, v) { var n = parseInt(v, 10); pl.draftedRound = (n > 0 ? n : null); });
    bindInlineInput(".kpr-yrs", function (pl, v) { var n = parseInt(v, 10); pl.yearsKept = (n > 0 ? n : 0); });
  }

  // Patch the derived columns of every row in place (matched by player id, so a
  // changed sort order doesn't matter) without touching the input cells. The
  // table keeps its current row order while you edit — it re-sorts on the next
  // full render (limit / rule change) so a row doesn't jump under your cursor.
  function patchTable(rows) {
    var mx = maxAbs(rows);
    var byPid = {};
    Array.prototype.forEach.call(elTbody.children, function (tr) { byPid[tr.getAttribute("data-pid")] = tr; });
    rows.forEach(function (row) {
      var tr = byPid[String(row.p.id)];
      if (!tr) return;
      var c = tr.querySelector(".kpr-c-cost"); if (c) c.textContent = "Round " + row.cost;
      var m = tr.querySelector(".kpr-c-mkt"); if (m) m.innerHTML = mktCell(row);
      var s = tr.querySelector(".kpr-c-surp"); if (s) s.innerHTML = surpCell(row, mx);
      var v = tr.querySelector(".kpr-c-verd"); if (v) v.innerHTML = verdictCell(row);
    });
  }

  // Live inline edit: update the model on every keystroke and re-price without
  // rebuilding the field, so nothing locks and a mistyped number is easy to fix.
  function bindInlineInput(sel, apply) {
    Array.prototype.forEach.call(elTbody.querySelectorAll(sel), function (inp) {
      inp.addEventListener("input", function () {
        var pl = players.filter(function (p) { return String(p.id) === String(inp.getAttribute("data-id")); })[0];
        if (!pl) return;
        apply(pl, inp.value);
        var rows = compute();
        patchTable(rows);
        renderOptimizer(rows);
      });
    });
  }

  function render() {
    var rows = compute();
    renderOptimizer(rows);
    renderTable(rows);
  }

  function showView(which) {
    var opt = which === "opt";
    if (elOpt) elOpt.hidden = !opt;
    if (elTbl) elTbl.hidden = opt;
    if (elViewOpt) elViewOpt.setAttribute("aria-selected", String(opt));
    if (elViewTbl) elViewTbl.setAttribute("aria-selected", String(!opt));
  }

  [elLim, elCost, elEsc, elUndr].forEach(function (el) { if (el) el.addEventListener("input", render); });
  if (elOpr) elOpr.addEventListener("change", render);
  if (elViewOpt) elViewOpt.addEventListener("click", function () { showView("opt"); });
  if (elViewTbl) elViewTbl.addEventListener("click", function () { showView("tbl"); });

  // Handoff to the Draft Room: stash the viewer's actual keeper picks so the
  // draft room overrides its projection for your team, then navigate. The draft
  // room computes the league-wide (projected) keeper set server-side.
  var elToDraft = $("kpr-to-draft");
  if (elToDraft && seed.draftUrl) {
    elToDraft.addEventListener("click", function () {
      var keptRows = compute().filter(function (r) { return r.keep; });
      var kept = keptRows.map(function (r) { return String(r.p.id); });
      // Carry each keeper's *resolved* cost round (after escalation + collision
      // bumps) so the draft room spends the right pick — the server recomputes
      // rival projections but can't know the per-player years-kept you entered.
      var keptDetail = keptRows.map(function (r) {
        return { id: String(r.p.id), costRound: r.cost, name: r.p.name, pos: r.p.pos };
      });
      var lim = parseInt(elLim && elLim.value, 10) || kept.length || 1;
      try {
        sessionStorage.setItem("brKeeperOverride", JSON.stringify({
          leagueId: String(seed.leagueId || ""),
          rosterId: String(seed.viewerRoster || ""),
          ids: kept,
          players: keptDetail
        }));
      } catch (e) { /* private mode: draft room still shows projections */ }
      // Carry the keeper rules so the other teams' projections use the same
      // ones you're playing by, instead of the server's defaults. The undrafted
      // cost matters most: left to the default, every player without a drafted
      // round (most of a dynasty roster) prices at the last round.
      var r = rules();
      var qs = "klimit=" + encodeURIComponent(lim) +
               "&kundr=" + encodeURIComponent(r.undraftedRound) +
               "&koff="  + encodeURIComponent(r.roundOffset) +
               "&kesc="  + encodeURIComponent(r.escalation) +
               "&kopr="  + (r.onePerRound ? "1" : "0");
      window.location.href = seed.draftUrl +
        (seed.draftUrl.indexOf("?") >= 0 ? "&" : "?") + qs;
    });
  }

  render();
})();
