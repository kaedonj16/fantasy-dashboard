// Rookie Draft Assistant — the Prospects-page Draft Board module, extracted
// from app.js so it only loads on the Prospects page instead of site-wide.
// Runs deferred, after app.js, so getCurrentRosterId and other globals exist.
// Exposes window._da and window.rkPageTab, used by the Prospects page markup.

(function () {
  let daProspects  = [];
  let daDrafted    = new Set(); // insertion order = overall draft pick order
  let myPicks      = new Set(); // subset of daDrafted that are the user's own picks
  let myPickOrder  = [];        // ordered subset of myPicks (for grading)
  let daLocalNeeds = {};        // position -> need level delta from my picks
  let daFilter    = 'ALL';
  let daSubView   = 'available'; // 'available' | 'drafted'
  let daNeeds      = {};
  let daLeagueType = '1qb';
  let daLeagueSize = 10;
  let daYear       = new Date().getFullYear();
  let daInitialized = false;

  const POS_COLORS = { QB: '#a78bfa', RB: '#34d399', WR: '#60a5fa', TE: '#fb923c' };
  const NEED_LABEL = { 2: 'Major Need', 1: 'Need', 0: 'Neutral', '-1': 'Depth', '-2': 'Stacked' };
  const NEED_COLOR = { 2: '#ef4444', 1: '#f59e0b', 0: '#9ca3af', '-1': '#10b981', '-2': '#059669' };
  const NEED_BONUS = { 2: 1.5, 1: 1.2, 0: 1.0, '-1': 0.85, '-2': 0.7 };

  function effectiveNeed(pos) {
    const raw   = daNeeds[pos] ?? 0;
    const delta = daLocalNeeds[pos] || 0;
    let need    = Math.max(-2, Math.min(2, raw + delta));
    // In 1QB leagues cap QB need at Neutral if roster already has 2+ QBs (including my picks)
    if (pos === 'QB' && daLeagueType !== 'sf') {
      const myQBs = myPickOrder.filter(id => {
        const p = daProspects.find(x => String(x.player_id) === id);
        return p && p.position === 'QB';
      }).length;
      if ((daNeeds.QB_count || 0) + myQBs >= 2) need = Math.min(-1, need);
    }
    return need;
  }

  function needBonus(pos) {
    return NEED_BONUS[String(effectiveNeed(pos))] ?? 1.0;
  }

  function adjustNeedsForDraft(playerId, delta) {
    const p = daProspects.find(x => String(x.player_id) === String(playerId));
    if (!p || !p.position) return;
    const pos = p.position.toUpperCase();
    daLocalNeeds[pos] = (daLocalNeeds[pos] || 0) + delta;
    renderNeeds();
  }

  function daScore(p) {
    const val = parseFloat(p.display_value || p.rookie_value || 0);
    return val * 0.6 + val * needBonus(p.position) * 0.4;
  }

  // 1 rec normally; 2 only if there's a major need AND the top pick doesn't address a need
  function recCount(scored) {
    const hasMajorNeed = Object.values(daNeeds).some(v => typeof v === 'number' && v === 2);
    if (!hasMajorNeed) return 1;
    const topNeed = scored[0] ? (daNeeds[scored[0].position] ?? 0) : 0;
    return topNeed >= 1 ? 1 : 2;
  }

  function daToggleNeeds() {
    const panel = document.getElementById('daNeedsPanel');
    if (!panel) return;
    const collapsed = panel.classList.toggle('da-needs-collapsed');
    const chevron = panel.querySelector('.da-needs-chevron');
    if (chevron) chevron.style.transform = collapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
  }

  function renderNeeds() {
    const panel = document.getElementById('daNeedsPanel');
    if (!panel) return;
    const collapsed = panel.classList.contains('da-needs-collapsed');
    const chevron = `<span class="da-needs-chevron" style="margin-left:auto;font-size:12px;transition:transform 0.2s;${collapsed?'transform:rotate(-90deg)':''}">&#8964;</span>`;
    const titleHtml = `<div class="da-needs-title" onclick="window._da.toggleNeeds()">My Roster Needs${chevron}</div>`;
    if (!Object.keys(daNeeds).length) {
      panel.innerHTML = titleHtml + '<div class="da-needs-body"><div style="font-size:12px;color:var(--text-muted);padding-top:8px;">Log in with your league to see personalized needs.</div></div>';
      return;
    }
    const rows = ['QB','RB','WR','TE'].map(pos => {
      const need  = effectiveNeed(pos);
      const col   = POS_COLORS[pos] || '#9ca3af';
      const count = daNeeds[`${pos}_count`] ?? 0;
      const val   = Math.round(daNeeds[`${pos}_value`] || 0);
      const avg   = Math.round(daNeeds[`${pos}_avg`]   || 0);
      return `<div class="da-need-row">
        <span class="pos-badge ${pos}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 7px;">${pos}</span>
        <div class="da-need-info">
          <span class="da-need-label" style="color:${NEED_COLOR[String(need)] || '#9ca3af'}">${NEED_LABEL[String(need)] ?? 'Neutral'}</span>
          <span class="da-need-meta">${count} players · ${val} (avg ${avg})</span>
        </div>
      </div>`;
    }).join('');
    panel.innerHTML = `${titleHtml}<div class="da-needs-body">${rows}</div>`;
  }

  function updateDraftedBadge() {
    const el = document.getElementById('daDraftedCount');
    if (!el) return;
    if (daDrafted.size === 0) { el.style.display = 'none'; }
    else { el.style.display = ''; el.textContent = daDrafted.size; }
  }

  function render() {
    const listEl = document.getElementById('daBoardList');
    if (!listEl) return;
    updateDraftedBadge();

    // Tag .da-board with current view so CSS can use different grid per view
    const boardEl = listEl.closest('.da-board');
    if (boardEl) boardEl.dataset.view = daSubView;

    if (daSubView === 'drafted') {
      // Sort by insertion order in daDrafted (first pick = index 0 = top)
      const draftedArr = [...daDrafted];
      const drafted = draftedArr
        .map(sid => daProspects.find(p => String(p.player_id) === sid))
        .filter(Boolean);
      if (!drafted.length) {
        listEl.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">No players drafted yet.</div>';
        return;
      }
      const endBtn = myPicks.size > 0
        ? `<div style="padding:12px 10px 4px;"><button class="da-end-draft-btn" onclick="window._da.endDraft()">End Draft &amp; Grade My Picks</button></div>`
        : '';
      listEl.innerHTML = endBtn + drafted.map((p, i) => {
        const sid   = String(p.player_id);
        const isMine = myPicks.has(sid);
        const col   = POS_COLORS[p.position] || '#9ca3af';
        const dAdp  = daLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
        const dTeam = p.actual_nfl_team || p.school || '';
        const dMeta = [dTeam, dAdp != null ? `ADP ${parseFloat(dAdp).toFixed(1)}` : ''].filter(Boolean).join(' · ');
        const overallPick = draftedArr.indexOf(sid) + 1;
        return `<div class="da-row${isMine ? ' da-my-pick' : ''}">
          <div class="da-rank"><span style="color:${isMine ? 'var(--accent)' : 'var(--text-muted)'};font-weight:${isMine ? '800' : '400'};">${overallPick}</span></div>
          <div class="da-info"><span class="da-name">${p.name || '–'}</span><span class="da-meta">${dMeta}</span></div>
          <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 6px;">${p.position}</span>
          <label class="da-mine-label" title="My pick">
            <input type="checkbox" class="da-mine-cb" ${isMine ? 'checked' : ''} onchange="window._da.toggleMine('${p.player_id}')">
            <span>Mine</span>
          </label>
          <div class="da-col-right da-val">${Math.round(parseFloat(p.display_value||0))||'–'}</div>
          <button class="otc-chip-remove" onclick="window._da.undraft('${p.player_id}')" title="Remove">×</button>
        </div>`;
      }).join('');
      return;
    }

    // Available view
    let visible = daProspects.filter(p => !daDrafted.has(String(p.player_id)));
    if (daFilter !== 'ALL') visible = visible.filter(p => p.position === daFilter);
    const scored = visible.map(p => ({ ...p, _s: daScore(p) })).sort((a, b) => b._s - a._s);
    const nRec   = recCount(scored);
    const recIds = new Set(scored.slice(0, nRec).map(p => String(p.player_id)));

    if (!scored.length) {
      listEl.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">No prospects available.</div>';
      return;
    }

    listEl.innerHTML = scored.map((p, i) => {
      const isRec    = recIds.has(String(p.player_id));
      const col      = POS_COLORS[p.position] || '#9ca3af';
      const val      = Math.round(parseFloat(p.display_value || 0));
      const needLvl  = daNeeds[p.position] ?? 0;
      const isNeed   = needLvl >= 1;
      const needCol  = NEED_COLOR[String(needLvl)] || '#9ca3af';
      const needTxt  = NEED_LABEL[String(needLvl)] || '';

      // Recommendation row: add grade + ADP in meta
      const adpRaw   = daLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
      const adpTxt   = adpRaw != null ? `ADP ${parseFloat(adpRaw).toFixed(1)}` : '';
      const gradeTxt = p.tier_label || '';
      const teamTxt  = p.actual_nfl_team || p.school || '';
      const baseMeta = [teamTxt, adpTxt].filter(Boolean).join(' · ');
      const recMeta  = isRec
        ? [teamTxt, gradeTxt, adpTxt].filter(Boolean).join(' · ')
        : baseMeta;

      // Need badge goes in the badge column (col 4) - same slot as PICK for rec rows
      const needBadge = isNeed && !isRec
        ? `<span style="font-size:10px;font-weight:700;color:${needCol};background:${needCol}18;border:1px solid ${needCol}33;border-radius:4px;padding:2px 6px;">${needTxt}</span>`
        : '';

      return `<div class="da-row${isRec ? ' da-recommended' : ''}">
        <div class="da-rank">${i + 1}</div>
        <div class="da-info">
          <span class="da-name">${p.name || '–'}${isRec && isNeed ? `<span style="font-size:10px;font-weight:700;color:${needCol};margin-left:6px;">▲ ${needTxt}</span>` : ''}</span>
          <span class="da-meta">${recMeta}</span>
        </div>
        <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 6px;">${p.position}</span>
        ${isRec ? '<div class="da-rec-badge">PICK</div>' : (needBadge || '<div></div>')}
        <div class="da-col-right da-val">${val || '–'}</div>
        <button class="da-draft-btn" onclick="window._da.draft('${p.player_id}')">Draft</button>
      </div>`;
    }).join('');
  }

  function saveSession() {
    const key = 'da_' + location.pathname;
    try {
      sessionStorage.setItem(key, JSON.stringify([...daDrafted]));
      sessionStorage.setItem(key + '_mine', JSON.stringify(myPickOrder));
    } catch (_) {}
  }

  function showDraftHelp() {
    const steps = [
      { icon: '1', title: 'Draft players in order', body: 'As each pick happens - yours or anyone else\'s - tap <strong>Draft</strong> to remove them from the board. Do this in real draft order so pick numbers are accurate.' },
      { icon: '2', title: 'Mark your picks', body: 'Switch to the <strong>Drafted</strong> tab and tap <strong>Mine</strong> on each player you actually selected. The pick number is set automatically based on when you drafted them.' },
      { icon: '3', title: 'Watch your needs update', body: 'The <strong>Roster Needs</strong> panel reflects your current roster vs. the league. Marking a pick as Mine adjusts the needs panel live.' },
      { icon: '4', title: 'End Draft &amp; grade', body: 'Once you\'ve marked your picks, tap <strong>End Draft &amp; Grade My Picks</strong>. Each pick is graded A+–F using ADP value, positional need, and QB context - the same formula as the Teams page Draft Grades.' },
    ];
    const html = `
      <div style="padding:20px 20px 0;display:flex;align-items:center;justify-content:space-between;">
        <div style="font-size:16px;font-weight:700;color:var(--text);">How to use the Draft Assistant</div>
        <button onclick="document.getElementById('daHelpModal').style.display='none'" style="background:none;border:none;font-size:20px;color:var(--text-muted);cursor:pointer;">✕</button>
      </div>
      <div style="padding:16px 20px 20px;display:flex;flex-direction:column;gap:16px;">
        ${steps.map(s => `
          <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;background:var(--accent);color:#fff;font-size:13px;font-weight:800;display:flex;align-items:center;justify-content:center;">${s.icon}</div>
            <div>
              <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:3px;">${s.title}</div>
              <div style="font-size:12px;color:var(--text-muted);line-height:1.5;">${s.body}</div>
            </div>
          </div>`).join('')}
      </div>`;

    let modal = document.getElementById('daHelpModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'daHelpModal';
      modal.style.cssText = 'display:none;position:fixed;inset:0;z-index:10600;align-items:center;justify-content:center;padding:20px;background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);';
      modal.innerHTML = '<div id="daHelpModalContent" style="background:var(--card);border-radius:16px;max-width:420px;width:100%;box-shadow:0 24px 48px rgba(15,23,42,0.25);"></div>';
      modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });
      document.body.appendChild(modal);
    }
    document.getElementById('daHelpModalContent').innerHTML = html;
    modal.style.display = 'flex';
  }

  // Exact port of pick_grade() and team_grade() from app.py (including BPA logic)
  function _pickGrade(adpDiff, need, pos, isSF, qbCount, numTeams, isBpa, bpaGap) {
    if (adpDiff === null) return 'N/A';
    const bigReach = -(numTeams * 1.1);
    let score;
    if      (adpDiff >= 4)          score = 4;
    else if (adpDiff >= 2)          score = 3;
    else if (adpDiff >= -3)         score = 2;
    else if (adpDiff >= bigReach)   score = 1;
    else                            score = 0;

    // BPA bonus / penalty (mirrors Python logic)
    if (isBpa) {
      score += adpDiff < -3 ? 1 : 2;
    } else if (bpaGap != null && bpaGap >= 5) {
      score = Math.max(score - 1, 0);
    }

    if (need) {
      score += 1;
    } else {
      if (pos === 'QB' && !isSF && qbCount >= 2) score = Math.max(score - 2, 0);
      else if (pos === 'QB' && !isSF && qbCount >= 1) score = Math.max(score - 1, 0);
    }
    if (adpDiff >= -3)            score = Math.max(score, 1);
    if (adpDiff >= 0)             score = Math.max(score, 2);  // positive value → min C
    if (need && adpDiff >= -4)    score = Math.max(score, 2);
    return ({5:'A+',4:'A',3:'B',2:'C',1:'D',0:'F'})[Math.min(score, 5)] || 'F';
  }

  function _teamGrade(grades) {
    if (!grades.length) return 'N/A';
    const v = {'A+':5,'A':4,'B':3,'C':2,'D':1,'F':0,'N/A':2};
    const avg = grades.reduce((s, g) => s + (v[g] ?? 2), 0) / grades.length;
    if (avg >= 4.5) return 'A+';
    if (avg >= 3.5) return 'A';
    if (avg >= 2.5) return 'B';
    if (avg >= 1.5) return 'C';
    if (avg >= 0.5) return 'D';
    return 'F';
  }

  function showDraftGrade() {
    const GRADE_COLOR = { 'A+': '#10b981', 'A': '#10b981', 'B': '#3b82f6', 'C': '#f59e0b', 'D': '#ef4444', 'F': '#ef4444', 'N/A': '#9ca3af' };
    const GRADE_BG    = { 'A+': 'rgba(16,185,129,.08)', 'A': 'rgba(16,185,129,.08)', 'B': 'rgba(59,130,246,.08)', 'C': 'rgba(245,158,11,.08)', 'D': 'rgba(239,68,68,.08)', 'F': 'rgba(239,68,68,.08)', 'N/A': 'transparent' };
    const isSF = daLeagueType === 'sf';

    const draftedArr = [...daDrafted]; // preserves insertion order = actual pick sequence

    // Build a lookup of adp for BPA computation
    const adpKey = p => parseFloat(isSF ? p.sf_avg_pick : p.avg_pick) || 9999;

    const picks = myPickOrder.map((sid, idx) => {
      const p = daProspects.find(x => String(x.player_id) === sid);
      if (!p) return null;
      const actualPick = draftedArr.indexOf(sid) + 1; // overall pick # in draft order
      const adp = parseFloat(isSF ? p.sf_avg_pick : p.avg_pick) || null;
      const adpDiff = adp !== null ? actualPick - adp : null;
      const need = (daNeeds[p.position] ?? 0) >= 1;
      const qbsBefore = myPickOrder.slice(0, idx).filter(id => {
        const q = daProspects.find(x => String(x.player_id) === id);
        return q && q.position === 'QB';
      }).length;
      const qbCount = (daNeeds.QB_count || 0) + qbsBefore;

      // BPA: who was available at this pick with a better ADP?
      const takenBefore = new Set(draftedArr.slice(0, actualPick - 1));
      const available = daProspects.filter(x => !takenBefore.has(String(x.player_id)));
      const bpa = available.reduce((best, x) => adpKey(x) < adpKey(best) ? x : best, available[0]);
      const bpaAdp = bpa ? adpKey(bpa) : null;
      const isBpa = bpa ? String(bpa.player_id) === sid : false;
      const bpaGap = (adp !== null && bpaAdp !== null && !isBpa) ? adp - bpaAdp : 0;

      const grade = _pickGrade(adpDiff, need, p.position, isSF, qbCount, daLeagueSize, isBpa, bpaGap);
      const needLabel = NEED_LABEL[String(daNeeds[p.position] ?? 0)] || 'Neutral';
      const tier = p.tier_label || '';
      return { p, actualPick, adp, adpDiff, grade, need, needLabel, tier, isBpa, bpaGap };
    }).filter(Boolean);

    if (!picks.length) return;

    const overall = _teamGrade(picks.map(x => x.grade));

    const rows = picks.map(({ p, actualPick, adp, adpDiff, grade, needLabel, tier, isBpa }) => {
      const col    = POS_COLORS[p.position] || '#9ca3af';
      const gc     = GRADE_COLOR[grade] || '#9ca3af';
      const gbg    = GRADE_BG[grade] || 'transparent';
      const adpTxt = adp ? `ADP ${adp.toFixed(1)}` : '';
      const pickTxt = `Pick ${actualPick}`;
      const diffTxt = isBpa
        ? 'Best player available'
        : adpDiff !== null && adpDiff < -1
          ? `${adpDiff.toFixed(1)} reach`
          : adpDiff !== null && adpDiff >= 0
            ? `+${adpDiff.toFixed(1)} value`
            : '';
      const diffCol = isBpa ? '#10b981' : adpDiff !== null ? (adpDiff >= 0 ? '#10b981' : '#ef4444') : 'var(--text-muted)';
      const tierTxt = tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : '';
      const bpaTxt  = isBpa ? '<span style="font-size:10px;font-weight:700;color:#10b981;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);border-radius:4px;padding:1px 5px;margin-left:4px;">BPA</span>' : '';
      const meta = [pickTxt, adpTxt, tierTxt].filter(Boolean).join(' · ');
      return `<div style="display:grid;grid-template-columns:1fr 38px 32px;align-items:center;gap:8px;padding:10px 14px 10px 12px;border-top:1px solid var(--border);border-left:3px solid ${gc};">
        <div style="display:flex;flex-direction:column;gap:2px;min-width:0;">
          <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
            <span style="font-size:13px;font-weight:700;color:var(--text);">${p.name}</span>${bpaTxt}
          </div>
          <span style="font-size:11px;color:var(--text-muted);">${meta}</span>
          ${diffTxt ? `<span style="font-size:11px;font-weight:600;color:${diffCol};">${diffTxt}</span>` : ''}
        </div>
        <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 5px;text-align:center;">${p.position}</span>
        <div style="font-size:18px;font-weight:800;color:${gc};text-align:right;">${grade}</div>
      </div>`;
    }).join('');

    const gc  = GRADE_COLOR[overall] || '#9ca3af';
    const gbg = GRADE_BG[overall] || 'transparent';
    const html = `
      <div style="padding:16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);">
        <div>
          <div style="font-size:15px;font-weight:700;color:var(--text);">My Draft Grade</div>
          <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">${picks.length} pick${picks.length !== 1 ? 's' : ''} graded</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:48px;height:48px;border-radius:50%;background:${gbg};border:2px solid ${gc};display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <span style="font-size:20px;font-weight:900;color:${gc};line-height:1;">${overall}</span>
          </div>
          <button onclick="document.getElementById('daGradeModal').style.display='none'" style="background:none;border:none;font-size:18px;color:var(--text-muted);cursor:pointer;padding:6px;line-height:1;flex-shrink:0;">✕</button>
        </div>
      </div>
      <div>${rows}</div>
      <div style="padding:14px 16px;display:flex;gap:8px;">
        <button onclick="document.getElementById('daGradeModal').style.display='none';daReset();" style="flex:1;padding:9px;background:transparent;color:var(--text-muted);border:1px solid var(--border);border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;">Reset Board</button>
        <button onclick="document.getElementById('daGradeModal').style.display='none'" class="da-end-draft-btn" style="flex:2;">Done</button>
      </div>`;

    let modal = document.getElementById('daGradeModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'daGradeModal';
      modal.style.cssText = 'display:none;position:fixed;inset:0;z-index:10600;align-items:center;justify-content:center;padding:20px;background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);';
      modal.innerHTML = '<div id="daGradeModalContent" style="background:var(--card);border-radius:16px;max-width:480px;width:100%;max-height:85vh;overflow-y:auto;box-shadow:0 24px 48px rgba(15,23,42,0.25);"></div>';
      modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });
      document.body.appendChild(modal);
    }
    document.getElementById('daGradeModalContent').innerHTML = html;
    modal.style.display = 'flex';
  }

  window._da = {
    draft(id)      { daDrafted.add(String(id));    saveSession(); render(); },
    undraft(id) {
      const sid = String(id);
      daDrafted.delete(sid);
      if (myPicks.has(sid)) {
        myPicks.delete(sid);
        myPickOrder = myPickOrder.filter(x => x !== sid);
        adjustNeedsForDraft(sid, +1);
      }
      saveSession(); render();
    },
    toggleMine(id) {
      const sid = String(id);
      if (myPicks.has(sid)) {
        myPicks.delete(sid);
        myPickOrder = myPickOrder.filter(x => x !== sid);
        adjustNeedsForDraft(sid, +1);
      } else {
        myPicks.add(sid);
        myPickOrder.push(sid);
        adjustNeedsForDraft(sid, -1);
      }
      saveSession(); render();
    },
    toggleNeeds()  { daToggleNeeds(); },
    endDraft()     { showDraftGrade(); },
    showHelp()     { showDraftHelp(); },
  };

  window.daFilterPos = function (pos) {
    daFilter = pos;
    document.querySelectorAll('.da-filter').forEach(b => b.classList.toggle('active', b.dataset.pos === pos));
    render();
  };

  window.daSubTab = function (sub) {
    daSubView = sub;
    document.querySelectorAll('.da-sub-tab').forEach(b => b.classList.toggle('active', b.dataset.sub === sub));
    render();
  };

  window.daReset = function () {
    daDrafted.clear();
    myPicks.clear();
    myPickOrder = [];
    daLocalNeeds = {};
    daFilter  = 'ALL';
    daSubView = 'available';
    document.querySelectorAll('.da-filter').forEach(b => b.classList.toggle('active', b.dataset.pos === 'ALL'));
    document.querySelectorAll('.da-sub-tab').forEach(b => b.classList.toggle('active', b.dataset.sub === 'available'));
    saveSession();
    render();
  };

  // Page-level tab switcher (Rankings / Draft Board)
  window.rkPageTab = function (tab) {
    document.querySelectorAll('.rk-page-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('rk-panel-rankings').style.display = tab === 'rankings' ? '' : 'none';
    document.getElementById('rk-panel-draft').style.display    = tab === 'draft'    ? '' : 'none';
    if (tab === 'draft' && !daInitialized) {
      daInitialized = true;
      initDA();
    }
    // Sync URL so refreshing/sharing lands on the same tab
    const _url = new URL(window.location.href);
    if (tab === 'draft') {
      _url.searchParams.set('tab', 'draft');
    } else {
      _url.searchParams.delete('tab');
    }
    history.replaceState(null, '', _url);
  };

  // Auto-open Draft Board tab when arriving via ?tab=draft link
  if (new URLSearchParams(window.location.search).get('tab') === 'draft') {
    rkPageTab('draft');
  }

  async function initDA() {
    const _sessKey = 'da_' + location.pathname;
    try { daDrafted = new Set(JSON.parse(sessionStorage.getItem(_sessKey) || '[]')); } catch (_) {}
    try { myPickOrder = JSON.parse(sessionStorage.getItem(_sessKey + '_mine') || '[]'); myPicks = new Set(myPickOrder); } catch (_) {}

    // Derive league context from URL: /<platform>/<season>/<league_id>/...
    const parts    = location.pathname.split('/').filter(Boolean);
    const platform = parts[0] || 'sleeper';
    const season   = parts[1] || new Date().getFullYear();
    const leagueId = parts[2];
    daYear         = parseInt(season);

    // Fetch the active draft class year (may differ from NFL season in URL)
    try {
      const acr = await fetch('/api/prospects/active-class');
      if (acr.ok) { const acd = await acr.json(); if (acd.year) daYear = acd.year; }
    } catch (_) {}

    // Fetch league-calibrated prospect rankings settings if in a league
    if (leagueId && !['players','breakouts','prospects','trade-database','trade-intel'].includes(platform)) {
      try {
        // Detect league type / size from rankings context (use rkLeagueType/rkLeagueSize if set by the Rankings tab)
        daLeagueType = (typeof rkLeagueType !== 'undefined' ? rkLeagueType : null)
          || localStorage.getItem('rk_league_type') || '1qb';
        daLeagueSize = parseInt((typeof rkLeagueSize !== 'undefined' ? rkLeagueSize : null)
          || localStorage.getItem('rk_league_size') || '10');

        // Get viewer roster_id if available on this page; backend falls back to session
        const viewerRid = (typeof getCurrentRosterId === 'function' ? getCurrentRosterId() : null)
          || document.querySelector('#viewerRosterIdInput')?.value || '';
        const needsUrl = `/api/draft-needs?league_id=${leagueId}&platform=${platform}&season=${season}`
          + (viewerRid ? `&roster_id=${encodeURIComponent(viewerRid)}` : '');
        const nr = await fetch(needsUrl);
        if (nr.ok) {
          const nd = await nr.json();
          if (nd.error) {
            const np = document.getElementById('daNeedsPanel');
            if (np) np.innerHTML = '<div class="da-needs-title">My Roster Needs<span class="da-needs-chevron" style="margin-left:auto;font-size:12px;">&#8964;</span></div><div class="da-needs-body"><div style="padding:12px 0;font-size:12px;color:var(--text-muted);">Log in with your league to see personalized needs.</div></div>';
          } else {
            daNeeds      = nd.needs || {};
            daLeagueType = nd.league_type || daLeagueType;
            daLeagueSize = nd.league_size || daLeagueSize;
          }
        }
      } catch (_) {}
    }

    renderNeeds();

    const listEl = document.getElementById('daBoardList');
    try {
      const r = await fetch(`/api/prospects/rankings?year=${daYear}&league_type=${encodeURIComponent(daLeagueType)}&league_size=${daLeagueSize}&limit=200`);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      daProspects = data.rankings || [];
      render();
    } catch (e) {
      if (listEl) listEl.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">Could not load prospects: ${e.message}</div>`;
    }
  }
})();
