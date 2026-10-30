// Player Rankings page module -- extracted verbatim from the inline <script>
// in page_players (app.py) so the browser caches and minifies it instead of
// re-downloading ~790 lines of inline JS on every rankings navigation.
// Loaded deferred, after app.js and after the inline window.__leagueTePremium
// injection, so both are available when this runs.

// [br-name-match:begin] Shared player-name matcher: ONE implementation for
// every player search surface (nav, trade calc, suggestions, compare,
// rankings, advanced metrics, trade intel, draft room, cheat sheet, dynasty
// tracker, portfolio). This block also lives verbatim in static/rankings.js,
// static/draft_room.js and static/cheat_sheet.js (those bundles can load
// without the app core); tests/test_shared_name_matcher.py pins the copies
// identical. Pure function declarations only (no top-level const/var/let),
// so loading two copies on one page can never collide. Mirrors the Python
// name_match_score in utils/players.py.
// Canonical normalize (same rules as Python normalize_name): lowercase,
// accents stripped, periods and apostrophes removed, hyphens become spaces,
// suffix tokens (jr/sr/ii/iii/iv/v) pulled out of the token list and kept
// separately. Score tiers: exact > substring > compact-equal (spaces
// removed: amonra, ceedee) > initials (JSN, CMC, BTJ) > compact substring >
// token-prefix in any order (jax sm, chase jamarr) > per-token typo via
// bounded Levenshtein. A suffix
// token in the query is a +25 ranking bonus when the player's suffix
// matches, never a hard filter. Callers sort score desc, then their own
// value desc, then name asc.
function brNameSuffixSet() {
  return { jr: 1, sr: 1, ii: 1, iii: 1, iv: 1, v: 1 };
}
function brNameNormTokens(name) {
  var s = String(name == null ? '' : name);
  try { s = s.normalize('NFD').replace(/[\u0300-\u036f]/g, ''); } catch (e) {}
  s = s.toLowerCase();
  s = s.replace(/[.''`]/g, '');
  s = s.replace(/[-‐-―]/g, ' ');
  s = s.replace(/[^a-z0-9]+/g, ' ');
  return s.split(' ').filter(Boolean);
}
function brNameCamelParts(tok) {
  // Split one original-case token at lower->upper boundaries:
  // McCaffrey -> [Mc, Caffrey], CeeDee -> [Cee, Dee]. Drives initials.
  var parts = [];
  var cur = '';
  for (var i = 0; i < tok.length; i++) {
    var ch = tok.charAt(i);
    if (i > 0 && ch >= 'A' && ch <= 'Z' && tok.charAt(i - 1) >= 'a' && tok.charAt(i - 1) <= 'z') {
      parts.push(cur);
      cur = ch;
    } else {
      cur += ch;
    }
  }
  if (cur) parts.push(cur);
  return parts;
}
function brNameTokenInit(origTok, normTok) {
  // Initials contributed by one token: "A.J." -> "aj", McCaffrey -> "mc",
  // CeeDee -> "cd", anything else -> its first letter.
  if (!origTok) return normTok ? normTok.charAt(0) : '';
  var cleaned = String(origTok).replace(/[^A-Za-z.]/g, '');
  if (cleaned.indexOf('.') >= 0) {
    var bits = cleaned.split('.').filter(Boolean);
    if (bits.length > 1) {
      return bits.map(function (b) { return b.charAt(0).toLowerCase(); }).join('');
    }
  }
  var parts = brNameCamelParts(cleaned);
  if (parts.length > 1) {
    return parts.map(function (p) { return p.charAt(0).toLowerCase(); }).join('');
  }
  return normTok ? normTok.charAt(0) : '';
}
function brNameFields(name) {
  var raw = String(name == null ? '' : name);
  var suffixSet = brNameSuffixSet();
  var rawToks = brNameNormTokens(raw);
  var suffix = '';
  var keep = [];
  for (var i = 0; i < rawToks.length; i++) {
    if (suffixSet[rawToks[i]]) suffix = rawToks[i];
    else keep.push(i);
  }
  var toks = keep.map(function (i) { return rawToks[i]; });
  if (!toks.length) { toks = rawToks.slice(); keep = rawToks.map(function (_, i) { return i; }); }
  // Original space/hyphen tokens align 1:1 with rawToks (normalizing only
  // strips punctuation inside a token), which drives per-token initials.
  var origToks = raw.split(/[\s\-‐-―]+/).filter(Boolean);
  var aligned = origToks.length === rawToks.length;
  var segInits = keep.map(function (idx) {
    return brNameTokenInit(aligned ? origToks[idx] : null, rawToks[idx]);
  });
  var initialsSet = {};
  var initials = [];
  function addInit(s) { if (s && !initialsSet[s]) { initialsSet[s] = 1; initials.push(s); } }
  var full = segInits.join('');
  addInit(full);
  if (suffix) addInit(full + suffix.charAt(0));
  var base = toks.map(function (t) { return t.charAt(0); }).join('');
  addInit(base);
  if (suffix) addInit(base + suffix.charAt(0));
  if (segInits.length > 1) {
    var fl = segInits[0] + segInits[segInits.length - 1];
    addInit(fl);
    if (suffix) addInit(fl + suffix.charAt(0));
  }
  return {
    norm: toks.join(' '),
    compact: toks.join(''),
    toks: toks,
    segInits: segInits,
    initials: initials,
    suffix: suffix
  };
}
function brNameParseQuery(query) {
  var rawToks = brNameNormTokens(query);
  var suffixSet = brNameSuffixSet();
  var suffix = '';
  var toks = [];
  for (var i = 0; i < rawToks.length; i++) {
    if (suffixSet[rawToks[i]]) suffix = rawToks[i];
    else toks.push(rawToks[i]);
  }
  return { norm: toks.join(' '), compact: toks.join(''), toks: toks, suffix: suffix };
}
function brNameLev(a, b, cap) {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > cap) return cap + 1;
  var prev = [];
  var cur = [];
  var i;
  var j;
  for (j = 0; j <= b.length; j++) prev[j] = j;
  for (i = 1; i <= a.length; i++) {
    cur[0] = i;
    var best = cur[0];
    for (j = 1; j <= b.length; j++) {
      var cost = a.charAt(i - 1) === b.charAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (cur[j] < best) best = cur[j];
    }
    if (best > cap) return cap + 1;
    var tmp = prev;
    prev = cur;
    cur = tmp;
  }
  return prev[b.length];
}
function brNameTokenOk(qt, fields, used) {
  for (var i = 0; i < fields.toks.length; i++) {
    if (used[i]) continue;
    if (fields.toks[i].indexOf(qt) === 0 || fields.segInits[i] === qt) {
      used[i] = 1;
      return true;
    }
  }
  return false;
}
function brNameTypoDist(qt, fields, used) {
  var cap = qt.length >= 6 ? 2 : (qt.length >= 4 ? 1 : 0);
  var bestI = -1;
  var bestD = cap + 1;
  for (var i = 0; i < fields.toks.length; i++) {
    if (used[i]) continue;
    var t = fields.toks[i];
    if (Math.abs(t.length - qt.length) > cap) continue;
    var d = brNameLev(qt, t, cap);
    if (d < bestD) { bestD = d; bestI = i; }
  }
  if (bestI < 0) return -1;
  used[bestI] = 1;
  return bestD;
}
function brScoreNameFields(fields, pq) {
  if (!fields || !pq || !pq.toks.length) return 0;
  var score = 0;
  if (fields.norm === pq.norm) {
    score = 1000;
  } else if (pq.norm && fields.norm.indexOf(pq.norm) >= 0) {
    score = 900 + (fields.norm.indexOf(pq.norm) === 0 ? 5 : 0);
  } else if (pq.compact && fields.compact === pq.compact) {
    score = 880;
  } else if (pq.toks.length === 1 && fields.initials.indexOf(pq.compact) >= 0) {
    score = 850;
  } else if (pq.compact.length >= 3 && fields.compact.indexOf(pq.compact) >= 0) {
    score = 840 + (fields.compact.indexOf(pq.compact) === 0 ? 5 : 0);
  } else {
    var used = {};
    var ok = true;
    var i;
    for (i = 0; i < pq.toks.length; i++) {
      if (!brNameTokenOk(pq.toks[i], fields, used)) { ok = false; break; }
    }
    if (ok) {
      score = 800;
    } else {
      used = {};
      ok = true;
      var dist = 0;
      for (i = 0; i < pq.toks.length; i++) {
        var d = brNameTypoDist(pq.toks[i], fields, used);
        if (d < 0) { ok = false; break; }
        dist += d;
      }
      if (ok) score = 700 - 10 * dist;
    }
  }
  if (score > 0 && pq.suffix && fields.suffix && pq.suffix === fields.suffix) score += 25;
  return score;
}
function brNameCachedFields(player, key, name) {
  var cached = player[key];
  if (cached && cached._for === name) return cached.f;
  var fields = brNameFields(name);
  try { player[key] = { _for: name, f: fields }; } catch (e) {}
  return fields;
}
function brScorePlayerName(player, query) {
  var pq = (query && typeof query === 'object' && query.toks) ? query : brNameParseQuery(query);
  if (!pq.toks.length) return 0;
  var isObj = player && typeof player === 'object';
  var name = isObj ? (player.name || '') : String(player == null ? '' : player);
  var searchName = isObj ? (player.search_name || '') : '';
  var best = 0;
  if (name) {
    var f = isObj ? brNameCachedFields(player, '_brNmF', name) : brNameFields(name);
    best = brScoreNameFields(f, pq);
  }
  if (searchName && searchName !== name) {
    var f2 = isObj ? brNameCachedFields(player, '_brNmS', searchName) : brNameFields(searchName);
    var s2 = brScoreNameFields(f2, pq);
    if (s2 > best) best = s2;
  }
  return best;
}
function brRankPlayersByName(players, query, valueFn, limit) {
  var pq = brNameParseQuery(query);
  if (!pq.toks.length) return [];
  var ranked = [];
  var list = players || [];
  for (var i = 0; i < list.length; i++) {
    var p = list[i];
    var s = brScorePlayerName(p, pq);
    if (s > 0) {
      ranked.push({ p: p, s: s, v: valueFn ? (Number(valueFn(p)) || 0) : 0 });
    }
  }
  ranked.sort(function (a, b) {
    if (b.s !== a.s) return b.s - a.s;
    if (b.v !== a.v) return b.v - a.v;
    var an = (a.p && a.p.name) || '';
    var bn = (b.p && b.p.name) || '';
    return an < bn ? -1 : (an > bn ? 1 : 0);
  });
  var out = ranked.map(function (x) { return x.p; });
  return limit ? out.slice(0, limit) : out;
}
if (typeof window !== 'undefined') {
  window.brNameMatch = {
    fields: brNameFields,
    parseQuery: brNameParseQuery,
    scoreFields: brScoreNameFields,
    score: brScorePlayerName,
    rank: brRankPlayersByName
  };
}
// [br-name-match:end]

// Dock the sticky rankings header just beneath the sticky top-nav. The nav's
// height changes with the viewport and PWA safe-area insets (and shrinks to just
// the logo on phones), so measure it rather than hardcode an offset.
(function prStickyOffset() {
  function setOffset() {
    var nav = document.querySelector('.top-nav');
    var h = nav ? Math.round(nav.getBoundingClientRect().height) : 56;
    document.documentElement.style.setProperty('--pr-sticky-top', h + 'px');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setOffset);
  } else {
    setOffset();
  }
  window.addEventListener('resize', setOffset);
})();

var prAllPlayers = [];
var prIndicators = {};
var prSparklines = {};
var prLeagueType   = (typeof brLeagueType === 'function' && brLeagueType()) || '1qb';
var prLeagueSize   = (typeof brLeagueSize === 'function' && brLeagueSize()) || 10;
var prScoringType  = (typeof window.__leagueScoringType === 'string' && window.__leagueScoringType)
  ? window.__leagueScoringType
  : ((typeof brScoringType === 'function' && brScoringType()) || 'dynasty');
// Positional rank (1-based, per position) for the ACTIVE view, keyed by player
// id. Recomputed each render from prGetValue() so it tracks the redraft/dynasty
// x 1qb/sf x size toggles instead of the dynasty-only precomputed *_pos_rank
// fields (e.g. an aging back is RB5 in redraft but RB12 in dynasty).
var prPosRankMap   = {};
// Default to the league's TE premium (injected from its scoring settings)
// and fall back to Off for the public/no-league view.
var prTePremium    = (typeof window.__leagueTePremium === 'number') ? window.__leagueTePremium : 0;
// Convert TE-premium points/reception into a value multiplier for TEs.
// ~+20% at full (1.0) PPR-TE premium, scaled linearly.
function prTeBoost(pos) {
  if (!prTePremium || pos !== 'TE') return 1;
  return 1 + prTePremium * 0.20;
}
// Reflect the league-derived TE premium in the settings toggle on load.
(function(){
  try {
    document.querySelectorAll('#prTepSection .settings-toggle').forEach(function(b){
      b.classList.toggle('active', Number(b.getAttribute('data-value')) === prTePremium);
    });
  } catch(e) {}
})();
var prPosFilters = new Set();   // empty = All
var prSearchQuery = '';
var prLoaded = false;
var _prLoadGen = 0;  // ignore stale fetch results if prLoadData re-enters
var prPage = 1;
var prPageSize = 50;
var prSortKey = 'value';   // active sort key (mirrors the #prSort dropdown)
var prSortDir = 'desc';    // 'asc' | 'desc' -- flipped by clicking a column header
// Default direction when a sort key is first chosen (AM-table convention:
// names, ages, ADPs and ranks ascend; values and point totals descend).
var PR_SORT_DIRS = { value: 'desc', adp: 'asc', age: 'asc', pos_rank: 'asc', ppg: 'desc', total_pts: 'desc', name: 'asc' };
// Sort control: choosing a key resets to that key's default direction.
function prSetSortKey(key) {
  prSortKey = key;
  prSortDir = PR_SORT_DIRS[key] || 'desc';
  var sel = document.getElementById('prSort');
  if (sel) sel.value = prSortKey;
  prSyncSortToggle();
  prPage = 1;
  prFlipRender();
}
// Sliding-pill sort (Teams page .teams-mode-toggle mechanism): the thumb
// measures the active button, exactly like _modeThumb on the Teams page.
function prSortThumb() {
  var wrap = document.getElementById('prSortToggle');
  if (!wrap) return;
  var thumb = wrap.querySelector('.teams-mode-thumb');
  var btn = wrap.querySelector('button.on');
  if (!thumb) return;
  if (!btn) { thumb.style.width = '0px'; return; }
  // Measure the active button (the _modeThumb mechanism from the Teams
  // page) with rect math instead of offsetWidth/offsetLeft: the offset
  // values are integer-rounded, which left the thumb up to 0.45px off
  // the button on fractional layouts.
  var wr = wrap.getBoundingClientRect();
  var br = btn.getBoundingClientRect();
  var borderL = parseFloat(getComputedStyle(wrap).borderLeftWidth) || 0;
  thumb.style.width = br.width + 'px';
  thumb.style.transform = 'translateX(' + (br.left - wr.left - borderL) + 'px)';
  wrap.classList.add('js-thumb');
}
function prSyncSortToggle() {
  var wrap = document.getElementById('prSortToggle');
  if (!wrap) return;
  wrap.querySelectorAll('button').forEach(function(b) {
    var on = b.getAttribute('data-sortkey') === prSortKey;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  prSortThumb();
}
// Header format segments (Dynasty / Redraft + 1QB / SF in Row 1) mirror the
// same state the Settings panel toggles carry.
function prSyncHeaderSegs() {
  document.querySelectorAll('#prScoringSeg .pr-segbtn').forEach(function(b) {
    b.classList.toggle('active', b.getAttribute('data-value') === prScoringType);
  });
  document.querySelectorAll('#prFormatSeg .pr-segbtn').forEach(function(b) {
    b.classList.toggle('active', b.getAttribute('data-value') === prLeagueType);
  });
}
// Legend panel toggle (Row 2, pinned right next to Settings).
function prToggleLegend() {
  var panel = document.getElementById('prLegendPanel');
  var btn = document.getElementById('prLegendBtn');
  if (!panel || !btn) return;
  var isOpen = panel.style.display === 'block';
  panel.style.display = isOpen ? 'none' : 'block';
  btn.classList.toggle('active', !isOpen);
  if (!isOpen) {
    var sp = document.getElementById('prSettingsPanel');
    var sb = document.getElementById('prSettingsBtn');
    if (sp) sp.style.display = 'none';
    if (sb) sb.classList.remove('active');
  }
}
// Headshot fallback chain: ESPN combiner (hi-res) -> raw ESPN URL -> Sleeper
// CDN -> initials disc. The disc stays hidden while any source loads.
function prImgFallback(img) {
  var fb = (img.getAttribute('data-fb') || '').split('|').filter(Boolean);
  if (fb.length) {
    img.setAttribute('data-fb', fb.slice(1).join('|'));
    img.src = fb[0];
  } else {
    img.style.display = 'none';
    var mono = img.nextElementSibling;
    if (mono) mono.style.display = 'flex';
  }
}
window.prImgFallback = prImgFallback;
// The shared ESPN combiner treatment (_hiResHeadshot in app.js, PR #2454).
// It lives in the features half of app.js, which lite (guest) pages lazy
// load, so fall back to a local mirror of the same transform (the same
// mirror app.py keeps as _hires_headshot_url) when it is not on the page.
function prHiResHeadshot(url, width) {
  if (typeof _hiResHeadshot === 'function') return _hiResHeadshot(url, width);
  if (!url || typeof url !== 'string') return url || '';
  if (url.indexOf('/combiner/') !== -1) return url;
  var m = url.match(/espncdn\.com(\/i\/headshots\/[^?]+\.(?:png|jpg|jpeg))/i);
  if (!m) return url;
  var w = Math.max(1, Math.round(width || 200));
  return 'https://a.espncdn.com/combiner/i?img=' + m[1] + '&w=' + w + '&scale=crop&cquality=100';
}
// One headshot cell for a row: the shared ESPN combiner treatment
// (_hiResHeadshot in app.js, PR #2454) with the Sleeper CDN as fallback and
// initials only when no photo exists or every source fails.
function prHeadshotHTML(p, isPick) {
  if (isPick) return '<span class="pr-photo"></span>';
  var name = String(p.name || '');
  var toks = name.split(/\s+/).filter(Boolean);
  var initials = toks.length
    ? (toks[0].charAt(0) + (toks.length > 1 ? toks[toks.length - 1].charAt(0) : '')).toUpperCase()
    : '?';
  var mono = '<span class="pr-mono" style="display:none" aria-hidden="true">' + initials + '</span>';
  var espn = String(p.espnHeadshot || '');
  var sleeper = p.id ? ('https://sleepercdn.com/content/nfl/players/thumb/' + encodeURIComponent(p.id) + '.jpg') : '';
  if (!espn && !sleeper) return '<span class="pr-photo">' + mono.replace('display:none', 'display:flex') + '</span>';
  // Combiner width 128: the cell displays 62px desktop / 50px phone, so
  // 128 is ~2x (retina-sharp) without shipping the 600px original.
  var hi = espn ? prHiResHeadshot(espn, 128) : '';
  var src = hi || sleeper;
  var fb = [];
  if (hi && espn && hi !== espn) fb.push(espn);
  if (sleeper && sleeper !== src) fb.push(sleeper);
  return '<span class="pr-photo"><img class="pr-hs" src="' + src + '" data-fb="' + fb.join('|') + '" onerror="prImgFallback(this)" alt="" loading="lazy" decoding="async" width="62" height="56">' + mono + '</span>';
}
// Clickable column headers (AM-table pattern): clicking the active key flips
// direction, clicking a new key takes its default direction. The dropdown is
// kept in sync so both controls always agree.
function prHeaderSort(key) {
  if (prSortKey === key) prSortDir = (prSortDir === 'desc' ? 'asc' : 'desc');
  else { prSortKey = key; prSortDir = PR_SORT_DIRS[key] || 'desc'; }
  var sel = document.getElementById('prSort');
  if (sel) sel.value = prSortKey;
  prSyncSortToggle();
  prPage = 1;
  prFlipRender();
}
// Ascending-base comparator for the active sort key. Multiplied by the sort
// direction so headers can flip asc/desc. Matches the old one-direction
// comparators when prSortDir is the key's default.
function prSortBase(a, b, sortBy) {
  if (sortBy === 'name') return String(a.name || '').localeCompare(String(b.name || ''));
  if (sortBy === 'age')  return (a.age != null ? a.age : 99) - (b.age != null ? b.age : 99);
  if (sortBy === 'adp') {
    var aA = prGetAdpSortVal(a), bA = prGetAdpSortVal(b);
    return (aA != null ? aA : 99999) - (bA != null ? bA : 99999);
  }
  if (sortBy === 'pos_rank') {
    var rA = prPosRankMap[String(a.id)] || 9999, rB = prPosRankMap[String(b.id)] || 9999;
    return rA - rB;
  }
  if (sortBy === 'ppg')       { var aP = a.ppg_raw != null ? a.ppg_raw : a.ppg, bP = b.ppg_raw != null ? b.ppg_raw : b.ppg; return (aP != null ? aP : -1) - (bP != null ? bP : -1); }
  if (sortBy === 'total_pts') { var aT = a.total_pts_raw != null ? a.total_pts_raw : a.total_pts, bT = b.total_pts_raw != null ? b.total_pts_raw : b.total_pts; return (aT != null ? aT : -1) - (bT != null ? bT : -1); }
  return prGetValue(a) - prGetValue(b);
}
function prSortCmp(a, b, sortBy) {
  // Missing values always sort last, in either direction (so a descending
  // age sort doesn't surface unknown-age players on top).
  var an = prSortIsNull(a, sortBy), bn = prSortIsNull(b, sortBy);
  if (an && bn) return 0;
  if (an) return 1;
  if (bn) return -1;
  return prSortBase(a, b, sortBy) * (prSortDir === 'asc' ? 1 : -1);
}
function prSortIsNull(p, sortBy) {
  if (sortBy === 'age')       return p.age == null;
  if (sortBy === 'adp')       return prGetAdpSortVal(p) == null;
  if (sortBy === 'pos_rank')  return !prPosRankMap[String(p.id)];
  if (sortBy === 'ppg')       return p.ppg == null;
  if (sortBy === 'total_pts') return p.total_pts == null;
  if (sortBy === 'value' || sortBy === 'rank') return !(prGetValue(p) > 0);
  return false; // 'name'
}
// Arrow on the active sort header (AM-table pattern). The ADP-source view
// replaces the header with its own sortable columns, so this no-ops there.
function prUpdateSortHeaders() {
  var header = document.getElementById('prTableHeader');
  if (!header || header.dataset.adpMode === '1') return;
  // The reworked header carries Player / PPG / Total Points / Value
  // columns; Age and Pos Rank sorts live on the sliding pill, which carries
  // their active state, so they get no column arrow.
  var col = null;
  if (prSortKey === 'name') col = 'player';
  else if (prSortKey === 'ppg') col = 'ppg';
  else if (prSortKey === 'total_pts') col = 'total_pts';
  else if (prSortKey === 'value' || prSortKey === 'rank') col = 'sort';
  header.querySelectorAll('[data-pr-sort-col]').forEach(function(el) {
    var on = el.getAttribute('data-pr-sort-col') === col;
    el.classList.toggle('sorted-asc', on && prSortDir === 'asc');
    el.classList.toggle('sorted-desc', on && prSortDir === 'desc');
    if (on) el.setAttribute('aria-sort', prSortDir === 'asc' ? 'ascending' : 'descending');
    else el.removeAttribute('aria-sort');
  });
}
var prAdpSourceOptions = {};    // {startup|rookie|redraft: [{value,label}]} from payload
var prAdpSources = {};          // {startup|rookie|redraft: 'Sleeper'|...} label the server used
var prAdpSource = 'auto';       // currently selected ADP source ('auto' = server default)
var prAdpReloading = false;     // guards concurrent source re-fetches
var prAdpColumns = [];          // [{value,label}] per-source ADP columns for the sort-by-ADP view
var prAdpSortSource = '';       // which source column the ADP view is sorted by ('' = default)
var PR_ADP_COL_W = 82;            // compact source columns; the table scrolls horizontally when every source is visible
var PR_ADP_COL_W_MOBILE = 76;     // touch-friendly source cols -- mobile scrolls instead of squeezing to fit
var PR_ADP_RANK_W_MOBILE = 40;    // sticky # column width on mobile (must match CSS left offset)
var PR_ADP_PLAYER_W_MOBILE = 132; // sticky Player column width on mobile

var PR_SPARK_W = 38, PR_SPARK_H = 26;  // logical (CSS) px
// Set true for the one render pass right after sparkline data first loads, so
// the lines draw on left-to-right; false for later sort/filter re-renders.
var _prSparkAnimate = false;
// Cached win/loss colors. getComputedStyle() forces a style recalc, so calling
// it per-sparkline (50x per render) was a major slowdown. Compute once.
var _prSparkColors = null;
function _prSparkColor(up) {
  if (!_prSparkColors) {
    var cs = getComputedStyle(document.documentElement);
    _prSparkColors = {
      win: (cs.getPropertyValue('--win') || '#16a34a').trim() || '#16a34a',
      loss: (cs.getPropertyValue('--loss') || '#ef4444').trim() || '#ef4444'
    };
  }
  return up ? _prSparkColors.win : _prSparkColors.loss;
}

function _prReducedMotion() {
  return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

// Stroke the polyline up to a fraction `progress` (0–1) of its length, so the
// caller can sweep it on. Even x-spacing means progress maps cleanly to segments.
function _prPaintSpark(ctx2d, pts, w, h, progress, col) {
  ctx2d.clearRect(0, 0, w, h);
  const n = pts.length, segs = n - 1;
  const t = progress * segs;
  const full = Math.max(0, Math.min(segs, Math.floor(t)));  // never index out of range
  // The polyline visible at this progress (for the reveal animation).
  const line = [pts[0]];
  for (let i = 1; i <= Math.min(full, segs); i++) line.push(pts[i]);
  let tip;
  if (full < segs) {
    const frac = t - full, a = pts[full], b = pts[full + 1];
    tip = { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
    line.push(tip);
  } else {
    tip = pts[n - 1];
  }
  // Filled area under the line.
  ctx2d.beginPath();
  ctx2d.moveTo(line[0].x, h);
  line.forEach(p => ctx2d.lineTo(p.x, p.y));
  ctx2d.lineTo(line[line.length - 1].x, h);
  ctx2d.closePath();
  ctx2d.globalAlpha = 0.15;
  ctx2d.fillStyle = col;
  ctx2d.fill();
  ctx2d.globalAlpha = 1;
  // The line itself.
  ctx2d.beginPath();
  ctx2d.moveTo(line[0].x, line[0].y);
  for (let i = 1; i < line.length; i++) ctx2d.lineTo(line[i].x, line[i].y);
  ctx2d.strokeStyle = col;
  ctx2d.lineWidth = 1.7;
  ctx2d.lineJoin = 'round';
  ctx2d.lineCap = 'round';
  ctx2d.stroke();
  // Endpoint dot at the current tip.
  ctx2d.beginPath();
  ctx2d.arc(tip.x, tip.y, 2.1, 0, Math.PI * 2);
  ctx2d.fillStyle = col;
  ctx2d.fill();
}

function _prDrawSparkline(canvas, data, animate, w, h) {
  if (!canvas || !data || data.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  w = w || PR_SPARK_W;
  h = h || PR_SPARK_H;
  // Size the backing store for the display density so the line is crisp.
  canvas.width  = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  const ctx2d = canvas.getContext('2d');
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx2d.clearRect(0, 0, w, h);
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pad = 3;   // room for the endpoint dot
  const pts = data.map((v, i) => ({
    x: pad + (i / (data.length - 1)) * (w - pad * 2),
    y: (h - pad) - ((v - min) / range) * (h - pad * 2)
  }));
  // Trend color from the app's own win/loss tokens (so it tracks the theme and
  // matches the rank arrows), green when the value is up over the window, red down.
  // Colors are cached module-wide (see _prSparkColor) to avoid a forced style
  // recalc per sparkline.
  const col = _prSparkColor(data[data.length - 1] >= data[0]);
  // A new draw (or a re-render that recycled this canvas) must cancel the
  // previous sweep; otherwise the old rAF keeps painting a half-drawn line
  // and the sparkline looks hung.
  canvas._prSparkGen = (canvas._prSparkGen || 0) + 1;
  const gen = canvas._prSparkGen;
  if (animate && !_prReducedMotion()) {
    let start = null; const dur = 650;
    requestAnimationFrame(function frame(now) {
      if (canvas._prSparkGen !== gen || canvas.isConnected === false) return;
      if (start === null) start = now;
      // Clamp to [0,1]: rAF timestamps vs performance.now() can invert, which
      // used to make p (and progress) negative → a crash that blanks the line.
      const p = Math.min(1, Math.max(0, (now - start) / dur));
      try {
        _prPaintSpark(ctx2d, pts, w, h, 1 - Math.pow(1 - p, 3), col); // easeOutCubic
      } catch (err) {
        _prPaintSpark(ctx2d, pts, w, h, 1, col);
        return;
      }
      if (p < 1) requestAnimationFrame(frame);
    });
  } else {
    _prPaintSpark(ctx2d, pts, w, h, 1, col);
  }
}

// Single source of truth for value column key selection (used by prGetValue + prGetSparkData)
function prValueKey(isSf) {
  if (isSf) return prLeagueSize === 10 ? 'sf_value' : 'sf_value_' + prLeagueSize;
  return prLeagueSize === 10 ? 'value' : 'value_' + prLeagueSize;
}

function prGetSparkData(pid) {
  const entry = prSparklines[pid];
  if (!entry) return null;
  if (Array.isArray(entry)) return entry.length >= 2 ? entry : null;
  if (prLeagueType === 'sf') {
    return entry[prValueKey(true)] || entry['sf_value'] || entry['value'] || null;
  }
  return entry[prValueKey(false)] || entry['value'] || null;
}

// Rescale a sparkline series so its last point equals `target` (the value shown
// in the row). The API scales each series to the calibrated base value, which
// omits the TE premium and size-specific column that prGetValue applies - so
// without this the sparkline's endpoint drifts from the displayed number.
function prAlignSpark(series, target) {
  if (!series || series.length < 2) return series;
  const last = series[series.length - 1];
  if (!last || !isFinite(last) || !isFinite(target) || target <= 0) return series;
  const scale = target / last;
  // Skip a no-op rescale (already aligned) to avoid tiny float churn.
  if (Math.abs(scale - 1) < 1e-6) return series;
  return series.map(v => v * scale);
}

// ---- Fuzzy search: the shared brNameMatch matcher (verbatim block at the
// top of this file; same implementation as the app core bundle) ----
function prFuzzyScore(name, query) {
  return (typeof brScorePlayerName === 'function') ? brScorePlayerName(name, query) : 0;
}

function prGetValue(p) {
  let base;
  if (prScoringType === 'redraft') {
    // Size-invariant: same 10-team columns the player modal shows.
    // redraft_sf_value_{8,12,14} is a trade-WLS overlay that has been
    // collapsing to a 1QB-shaped board (QB1 ~450, overall #30+).
    base = prLeagueType === 'sf'
      ? Number(p.redraft_value_sf ?? p.redraft_value_1qb ?? 0)
      : Number(p.redraft_value_1qb ?? 0);
  } else if (prLeagueType === 'sf') {
    base = Number(p[prValueKey(true)] ?? p.sf_value ?? p.value ?? 0);
  } else {
    base = Number(p[prValueKey(false)] ?? p.value ?? 0);
  }
  base *= prTeBoost(p.position);
  return Math.round(base * 10) / 10;
}

function prFormatValue(v) {
  if (!v || v <= 0) return '-';
  return v.toFixed(1);
}

// POS-cell label for the active view. Uses the per-render prPosRankMap (ranked by
// prGetValue) so it reflects the redraft/dynasty x 1qb/sf x size toggles; falls
// back to the dynasty precomputed labels only before the first render populates it.
function prPosRankLabel(p) {
  const n = prPosRankMap[String(p.id)];
  if (n) return (p.position || '') + n;
  return prLeagueType === 'sf'
    ? (p.sf_pos_rank_label || p.pos_rank_label || p.position)
    : (p.pos_rank_label || p.position);
}

function prIsRookie(id) {
  return prIndicators.rookies && prIndicators.rookies.includes(String(id));
}

function prIsBreakout(id) {
  return prIndicators.breakouts && prIndicators.breakouts.includes(String(id));
}

function prIsElite(id) {
  return prIndicators.elites && prIndicators.elites.includes(String(id));
}

function prIsProspect(id) {
  return prIndicators.prospects && prIndicators.prospects.includes(String(id));
}

// Settings panel toggle
function prToggleSettings() {
  const panel = document.getElementById('prSettingsPanel');
  const btn = document.getElementById('prSettingsBtn');
  if (!panel || !btn) return;

  const isOpen = panel.style.display === 'block';
  if (isOpen) {
    panel.style.display = 'none';
    btn.classList.remove('active');
  } else {
    panel.style.display = 'block';
    btn.classList.add('active');
    const lp = document.getElementById('prLegendPanel');
    const lb = document.getElementById('prLegendBtn');
    if (lp) lp.style.display = 'none';
    if (lb) lb.classList.remove('active');
    if (window.innerWidth <= 768) {
      const rect = btn.getBoundingClientRect();
      panel.style.position = 'fixed';
      panel.style.top = (rect.bottom + 6) + 'px';
      panel.style.left = '12px';
      panel.style.right = '12px';
      panel.style.minWidth = '';
    } else {
      panel.style.position = '';
      panel.style.top = '';
      panel.style.left = '';
      panel.style.right = '';
      panel.style.minWidth = '';
    }
  }
}

// Update active settings indicator tags
function updateSettingsIndicator() {
  const indicator = document.getElementById('prActiveSettings');
  if (!indicator) return;

  const tags = indicator.querySelectorAll('.active-setting-tag');
  if (tags[0]) tags[0].textContent = prLeagueSize + '-Team';
  if (tags[1]) tags[1].textContent = prLeagueType.toUpperCase();
  if (tags[2]) tags[2].textContent = prScoringType === 'redraft' ? 'Redraft' : 'Dynasty';
  const tepTag = document.getElementById('prTepTag');
  if (tepTag) {
    tepTag.style.display = prTePremium ? '' : 'none';
    tepTag.textContent = 'TE+' + prTePremium;
  }
}

function prSetScoringType(type) {
  prScoringType = type;
  var sub = document.getElementById('prSubtitle');
  if (sub) {
    sub.textContent = type === 'redraft'
      ? 'All players ranked by this-season redraft value.'
      : 'All players ranked by dynasty value.';
  }
  // Update panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Scoring')) {
      btn.classList.toggle('active', btn.getAttribute('data-value') === type);
    }
  });
  // Hide league-size in redraft (size doesn't affect redraft values)
  const sizeSection = document.getElementById('prSizeSection');
  if (sizeSection) sizeSection.style.display = type === 'redraft' ? 'none' : '';
  // Hide PICK and ROOKIE filters in redraft
  document.querySelectorAll('.pos-pill[data-pos="PICK"], .pos-pill[data-pos="ROOKIE"]').forEach(btn => {
    btn.style.display = type === 'redraft' ? 'none' : '';
  });
  if (type === 'redraft' && (prPosFilters.has('PICK') || prPosFilters.has('ROOKIE'))) {
    prPosFilters.clear();
  }
  prSyncHeaderSegs();
  updateSettingsIndicator();
  prPage = 1;
  if (prLoaded) prRender();
}

function prSetLeagueType(type) {
  prLeagueType = type;

  // Update settings panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Format')) {
      btn.classList.toggle('active', btn.getAttribute('data-value') === type);
    }
  });

  prSyncHeaderSegs();
  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

function prSetTePremium(pts) {
  prTePremium = Number(pts) || 0;
  // Highlight the active TE-premium button
  const sec = document.getElementById('prTepSection');
  if (sec) {
    sec.querySelectorAll('.settings-toggle[data-value]').forEach(btn => {
      btn.classList.toggle('active', Number(btn.getAttribute('data-value')) === prTePremium);
    });
  }
  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

function prSetSize(size) {
  prLeagueSize = size;

  // Update settings panel toggles
  document.querySelectorAll('#prSettingsPanel .settings-toggle[data-value]').forEach(btn => {
    const section = btn.closest('.settings-section');
    if (section && section.querySelector('.settings-section-label').textContent.includes('Size')) {
      const btnSize = parseInt(btn.getAttribute('data-value'));
      btn.classList.toggle('active', btnSize === size);
    }
  });

  updateSettingsIndicator();
  prPage = 1;
  prRender();
}

// Multi-select position toggle
function prTogglePos(pos) {
  if (pos === 'ALL') {
    prPosFilters.clear();
  } else {
    if (prPosFilters.has(pos)) {
      prPosFilters.delete(pos);
    } else {
      prPosFilters.add(pos);
    }
  }
  // Sync button states
  document.querySelectorAll('.pos-pill').forEach(b => {
    const p = b.getAttribute('data-pos');
    const on = (p === 'ALL') ? (prPosFilters.size === 0) : prPosFilters.has(p);
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
  prPage = 1;
  prRender();
}

function prClearSearch() {
  document.getElementById('prSearch').value = '';
  prSearchQuery = '';
  document.getElementById('prSearchClear').style.display = 'none';
  prPage = 1;
  prRender();
}

// Build overall rank map keyed by player id (ranked by current value)
function prBuildRankMap() {
  const ranked = prAllPlayers
    .filter(p => p.position !== 'PICK')
    .slice()
    .sort((a, b) => prGetValue(b) - prGetValue(a));
  return new Map(ranked.map((p, i) => [String(p.id), i + 1]));
}

// Map sort key → { header label, cell value function }
// ADP for the current scoring mode (redraft vs dynasty/startup) and league type
// (SF vs 1QB). The server attaches both a dynasty (avg_pick/sf_avg_pick) and a
// redraft (redraft_avg_pick/sf_redraft_avg_pick) ADP per player; pick the pair
// matching the Redraft/Dynasty toggle. Null ADP renders as a dash and sorts last.
function prGetAdp(p) {
  if (prScoringType === 'redraft') {
    return prLeagueType === 'sf' ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
  }
  return prLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
}
function prFormatAdp(p) {
  const v = prGetAdp(p);
  return (v != null) ? Number(v).toFixed(1) : '–';
}

// Which ADP source menu applies, given the current scoring mode.
function prAdpMode() {
  return prScoringType === 'redraft' ? 'redraft' : 'startup';
}

// ── Per-source ADP columns (the "sort by ADP" view) ─────────────────────────
// The ADP field on a player object for the current Dynasty/Redraft + 1QB/SF axis.
function prAdpField() {
  if (prScoringType === 'redraft') return prLeagueType === 'sf' ? 'sf_redraft_avg_pick' : 'redraft_avg_pick';
  return prLeagueType === 'sf' ? 'sf_avg_pick' : 'avg_pick';
}
// One source's ADP for a player on the current axis (null when that source has none).
function prAdpSourceVal(p, src) {
  const bs = p.adp_by_source && p.adp_by_source[src];
  if (!bs) return null;
  const v = bs[prAdpField()];
  return (v != null) ? Number(v) : null;
}
// Whether we have per-source columns to show (server sent them for this pool).
function prHasAdpColumns() { return Array.isArray(prAdpColumns) && prAdpColumns.length > 0; }
// Per-source columns that actually carry a value on the CURRENT axis. ESPN/Yahoo/
// MFL are redraft-only, so on the dynasty/rookie axes they hold no value and would
// render as an all-dashes column -- hide them instead of showing an empty column.
function prVisibleAdpColumns() {
  if (!prHasAdpColumns()) return [];
  const field = prAdpField();
  return prAdpColumns.filter(c =>
    prAllPlayers.some(p => {
      const bs = p.adp_by_source && p.adp_by_source[c.value];
      return bs && bs[field] != null;
    }));
}
// The source the ADP view is currently sorted by (defaults to Consensus, else
// first) -- chosen among the columns visible on the current axis.
function prActiveAdpSource() {
  const cols = prVisibleAdpColumns();
  if (!cols.length) return '';
  if (prAdpSortSource && cols.some(c => c.value === prAdpSortSource)) return prAdpSortSource;
  const cons = cols.find(c => c.value === 'consensus');
  return cons ? cons.value : cols[0].value;
}
// ADP value used for sorting rows: the active source's when columns exist,
// otherwise the legacy single-source field.
function prGetAdpSortVal(p) {
  if (prHasAdpColumns()) return prAdpSourceVal(p, prActiveAdpSource());
  return prGetAdp(p);
}
// Header click: sort the ADP view by a chosen source column.
function prSortByAdpSource(src) {
  prAdpSortSource = src;
  prPage = 1;
  if (typeof prFlipRender === 'function') prFlipRender(); else prRender();
}
window.prSortByAdpSource = prSortByAdpSource;

// Show the ADP-source dropdown only while sorting by ADP, and populate it with
// the sources valid for the current scoring mode (Yahoo redraft-only, BR
// Fantasy dynasty/rookie only). Preserves the current selection when possible.
function prSyncAdpSourceUI(sortBy) {
  const wrap = document.getElementById('prAdpSrcWrap');
  const sel = document.getElementById('prAdpSource');
  if (!wrap || !sel) return;
  // The per-source columns replace the single-source dropdown entirely; only
  // fall back to the dropdown when the server didn't send columns.
  const show = sortBy === 'adp' && !prHasAdpColumns();
  wrap.style.display = show ? '' : 'none';
  if (!show) return;
  const mode = prAdpMode();
  // "Auto" is the default the page loads with (the server's memoized attach).
  // Label it with whatever source the server actually used, so the dropdown
  // never claims a source the shown ADP didn't come from.
  const usedLabel = prAdpSources[mode];
  const autoLabel = usedLabel && usedLabel !== 'none' ? ('Auto (' + usedLabel + ')') : 'Auto';
  let opts = [{ value: 'auto', label: autoLabel }].concat(prAdpSourceOptions[mode] || []);
  // Yahoo ADP needs a connected Yahoo league (league id + token); hide it
  // anywhere else so the menu never offers a source that can't return data here.
  const yahooAvailable = (typeof window.__platform !== 'undefined'
                          && window.__platform === 'yahoo' && !!window.__leagueId);
  if (!yahooAvailable) opts = opts.filter(o => o.value !== 'yahoo');
  const want = sel.value || prAdpSource;
  sel.innerHTML = opts.map(o =>
    '<option value="' + o.value + '">' + o.label + '</option>').join('');
  const has = opts.some(o => o.value === want);
  sel.value = has ? want : 'auto';
  prAdpSource = sel.value;
}

// Swap the table header between the normal columns and the ADP-source columns.
// Caches the default header once so it can be restored when leaving the ADP view.
function prSetupAdpHeader(adpView, cols, active, gridCols, extra) {
  const header = document.getElementById('prTableHeader');
  if (!header) return;
  if (header._prDefaultHTML == null) {
    header._prDefaultHTML = header.innerHTML;
    header._prDefaultCols = header.style.gridTemplateColumns;
  }
  header.classList.toggle('pr-adp-mode', !!adpView);
  if (adpView) {
    header.style.gridTemplateColumns = gridCols;
    const metaHead = extra
      ? '<span class="pr-adp-meta-h">Pos</span><span class="pr-adp-meta-h">Age</span><span class="pr-adp-meta-h">Team</span>'
      : '';
    header.innerHTML =
      '<span class="pr-adp-pin pr-adp-pin-rank">#</span>' +
      '<span class="pr-adp-pin pr-adp-pin-player">Player</span>' +
      metaHead +
      cols.map(function (c) {
        const on = c.value === active;
        return '<span class="pr-adp-head' + (on ? ' pr-adp-head-active' : '') + '"' +
          ' role="button" tabindex="0" title="Sort by ' + c.label + ' ADP"' +
          ' onclick="prSortByAdpSource(\'' + c.value + '\')"' +
          ' onkeydown="if(event.key===\'Enter\'||event.key===\' \'){event.preventDefault();prSortByAdpSource(\'' + c.value + '\');}">' +
          c.label + (on ? ' <span class="pr-adp-sort-caret">▲</span>' : '') +
          '</span>';
      }).join('');
    header.dataset.adpMode = '1';
  } else if (header.dataset.adpMode === '1') {
    header.innerHTML = header._prDefaultHTML;
    header.style.gridTemplateColumns = header._prDefaultCols || '';
    header.dataset.adpMode = '0';
  }
}

// Re-fetch the pool with the chosen ADP source and merge only the ADP fields
// back onto the loaded players (preserving all client-side enrichment), then
// re-render. A no-op when nothing changed avoids a needless round trip.
function prReloadAdpSource() {
  const sel = document.getElementById('prAdpSource');
  if (!sel || prAdpReloading) return;
  prAdpSource = sel.value;
  // "Auto" re-fetches the server default (no adp_source override); any real
  // source overlays via the resolver.
  let url = '/api/league-players';
  if (prAdpSource && prAdpSource !== 'auto') {
    url += '?adp_source=' + encodeURIComponent(prAdpSource);
    if (window.__leagueId) url += '&league_id=' + encodeURIComponent(window.__leagueId);
    if (window.__platform) url += '&platform=' + encodeURIComponent(window.__platform);
  }
  prAdpReloading = true;
  fetch(url, { cache: 'no-store' })
    .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(resp => {
      const pls = Array.isArray(resp) ? resp : (resp.players || []);
      if (!Array.isArray(resp) && resp.adp_sources) prAdpSources = resp.adp_sources;
      const byId = {};
      pls.forEach(p => { byId[String(p.id)] = p; });
      const F = ['avg_pick', 'sf_avg_pick', 'rookie_avg_pick',
                 'sf_rookie_avg_pick', 'redraft_avg_pick', 'sf_redraft_avg_pick'];
      prAllPlayers.forEach(p => {
        const src = byId[String(p.id)];
        if (src) F.forEach(f => { p[f] = (src[f] != null ? src[f] : null); });
      });
      prPage = 1;
      prRender();
    })
    .catch(() => { /* keep current ADP on failure */ })
    .finally(() => { prAdpReloading = false; });
}

// var (not const) so this module can re-execute on a mobile soft-nav without a
// "PR_SORT_META has already been declared" error (const globals can't redeclare).
var PR_SORT_META = {
  rank:      { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
  value:     { label: 'Value',    cell: p => prFormatValue(prGetValue(p)) },
  age:       { label: 'Age',      cell: p => p.age != null ? Number(p.age).toFixed(1) : '–' },
  pos_rank:  { label: 'Pos Rank', cell: p => prPosRankLabel(p) },
  ppg:       { label: 'PPG',       cell: p => p.ppg != null ? p.ppg.toFixed(1) : '–' },
  total_pts: { label: 'Total Pts', cell: p => p.total_pts != null ? p.total_pts.toFixed(1) : '–' },
  adp:       { label: 'ADP',       cell: p => prFormatAdp(p) },
};

// Re-render with a FLIP animation so rows glide to their new positions when the
// sort order changes (used by the Sort dropdown). Falls back to a plain render
// when the helper is unavailable or the user prefers reduced motion.
function prFlipRender() {
  var list = document.getElementById('prList');
  var sortBy = prSortKey;
  // The ADP-source view rebuilds each row's grid (and often the overflow
  // scroller). FLIP-ing that layout change leaves rows translated inside the
  // clip -- the reorder that "gets stuck". Skip motion there; same-layout
  // sorts (Value / Age / PPG / …) still glide.
  var adpLayout = sortBy === 'adp' || (list && list.querySelector('.pr-adp-mode'));
  if (!adpLayout && list && window.brFlipReorder) window.brFlipReorder(list, prRender);
  else prRender();
}

// Sort and filter players, then render rows into the main table
function prRender() {
  if (!prLoaded) return;
  prQlHide();  // a re-render replaces the rows; any open quick-look is stale
  const sortBy = prSortKey;
  prSyncAdpSourceUI(sortBy);

  // "Sort by ADP" with per-source columns is a distinct view: the right-side
  // columns (Value/Age/Team/Pos) are swapped out for one column per ADP source.
  const isMobile = window.innerWidth <= 768;
  const adpView = (sortBy === 'adp') && prHasAdpColumns();
  const adpCols = adpView ? prVisibleAdpColumns() : [];
  const adpActive = adpView ? prActiveAdpSource() : '';
  // On desktop keep the Pos / Age / Team columns (compact) alongside the source
  // columns; on mobile drop them and pin # / Player while the source columns
  // scroll horizontally (same pattern as schedule rankings).
  const adpExtra = adpView && !isMobile;
  // Fixed-width source columns (slim, like Pos/Age/Team) instead of 1fr, so they
  // don't stretch across the row -- the Player column absorbs the slack on
  // desktop. On mobile, # + Player are fixed-width sticky pins and sources
  // scroll in #prTableScroll (see .pr-adp-pin-* in players_page.py).
  const _adpColW = (isMobile ? PR_ADP_COL_W_MOBILE : PR_ADP_COL_W) + 'px';
  const _adpSrcTracks = adpCols.map(() => _adpColW).join(' ');
  const ADP_GRID = adpView
    ? (adpExtra
        ? ('54px minmax(0,1fr) 40px 42px 46px ' + _adpSrcTracks)
        : (isMobile
            ? (PR_ADP_RANK_W_MOBILE + 'px ' + PR_ADP_PLAYER_W_MOBILE + 'px ' + _adpSrcTracks)
            : ('54px minmax(0,1fr) ' + _adpSrcTracks)))
    : '';
  const tableScroll = document.getElementById('prTableScroll');
  if (tableScroll) tableScroll.classList.toggle('pr-adp-scroll', adpView);
  prSetupAdpHeader(adpView, adpCols, adpActive, ADP_GRID, adpExtra);
  if (!adpView) {
    // On mobile (≤768px) the Age column is hidden, so switch the sort column
    // to show whatever is being sorted. On desktop all columns are visible.
    // PPG and Total Points are the exceptions: PPG has its own always-visible
    // column now, so when sorting by either the trailing column stays on Value.
    const _alwaysShowSort = sortBy === 'ppg' || sortBy === 'total_pts' || sortBy === 'adp';
    const sortMeta0 = (isMobile || _alwaysShowSort) ? (PR_SORT_META[sortBy] || PR_SORT_META.rank) : PR_SORT_META.rank;
    const headerMeta0 = (sortBy === 'ppg' || sortBy === 'total_pts') ? PR_SORT_META.rank : sortMeta0;
    const sortHeaderEl = document.getElementById('prSortHeader');
    if (sortHeaderEl) sortHeaderEl.textContent = headerMeta0.label;
    // Phone (approved mock): sorting by Total Points makes the total the
    // trailing column's big number (Value drops to a sublabel), so the
    // header names it. Desktop keeps Value on top and the header on Value.
    if (sortHeaderEl && isMobile && sortBy === 'total_pts') sortHeaderEl.textContent = 'Total';
    // Hide age col only on mobile when sort=age (shown in sort col instead)
    const ageHeaderEl = document.getElementById('prAgeHeader');
    if (isMobile && ageHeaderEl) ageHeaderEl.style.visibility = sortBy === 'age' ? 'hidden' : '';
    const ageColEls = document.querySelectorAll('.pr-age');
    if (isMobile) ageColEls.forEach(el => el.style.visibility = sortBy === 'age' ? 'hidden' : '');
    else ageColEls.forEach(el => el.style.visibility = '');
    // Arrow on the active sortable column header.
    prUpdateSortHeaders();
  }
  const _alwaysShowSort = sortBy === 'ppg' || sortBy === 'total_pts' || sortBy === 'adp';
  const sortMeta = (isMobile || _alwaysShowSort) ? (PR_SORT_META[sortBy] || PR_SORT_META.rank) : PR_SORT_META.rank;
  // The trailing column's metric. Sorting by PPG or Total Points no longer
  // swaps Value out: PPG renders in its own column and Total Points rides
  // along as a sublabel under Value, so the trailing column falls back to
  // Value for both sorts.
  const valueMeta = (sortBy === 'ppg' || sortBy === 'total_pts') ? PR_SORT_META.rank : sortMeta;

  let players = prAllPlayers.slice();

  // Positional rank for the ACTIVE view: rank every real player within his
  // position by the same value the VALUE column shows (prGetValue), so the POS
  // cell agrees with the ordering on screen. Computed over the full pool (not the
  // filtered/searched subset) so ranks are stable regardless of what's visible.
  prPosRankMap = {};
  {
    const _byPos = {};
    prAllPlayers.forEach(p => {
      if (p.position === 'PICK') return;
      if (p.is_rookie && !(p.team && p.team !== 'FA')) return;  // skip undrafted rookies
      if (!(prGetValue(p) > 0)) return;                          // no value in this view
      (_byPos[p.position] = _byPos[p.position] || []).push(p);
    });
    Object.keys(_byPos).forEach(pos => {
      _byPos[pos].sort((a, b) => prGetValue(b) - prGetValue(a));
      _byPos[pos].forEach((p, i) => { prPosRankMap[String(p.id)] = i + 1; });
    });
  }

  // In redraft mode exclude draft picks (not real players). Rookies stay --
  // they play this season, so a rookie with a redraft value (e.g. a first-round
  // RB) belongs on the board. The value check below drops any rookie the model
  // has no redraft value for, so we only show ones that actually rank.
  if (prScoringType === 'redraft') {
    players = players.filter(p => {
      if (p.position === 'PICK') return false;
      return prGetValue(p) > 0;
    });
  }

  const isDrafted = p => p.is_rookie && p.team && p.team !== 'FA';

  // Rank numbers ("#") are OVERALL ranks, computed from the full eligible pool
  // BEFORE the position filter (using the current sort). So filtering to a
  // position keeps each player on their overall rank (e.g. TE McBride shows his
  // overall #, not #1) -- which is what the overall movement arrow beside the #
  // measures, so the two line up -- and #s stay stable under search too.
  const _rankSort = (a, b) => prSortCmp(a, b, sortBy);
  const _rankMap = new Map();
  {
    let _rankIdx = 0;
    players.slice()
      .filter(p => p.position !== 'PICK' && !(p.is_rookie && !isDrafted(p)))
      .sort(_rankSort)
      .forEach(p => {
        // PPG / Total Points sorts show the player's ACTUAL overall rank
        // (server-computed on unrounded values over the full stat-line
        // pool, the same rank the player modal shows), not a row count of
        // this table. Other sorts keep the sequential board rank.
        const _srvRank = sortBy === 'ppg' ? p.ppg_ovr_rank
          : (sortBy === 'total_pts' ? p.total_pts_ovr_rank : null);
        if (_srvRank != null) {
          _rankMap.set(String(p.id), _srvRank);
          if (_srvRank > _rankIdx) _rankIdx = _srvRank;
        } else {
          _rankMap.set(String(p.id), ++_rankIdx);
        }
      });
  }

  // Position filter (multi-select) -- narrows which rows show, but each keeps the
  // overall rank number computed above.
  if (prPosFilters.has('ROOKIE')) {
    players = players.filter(p => p.is_rookie);
  } else if (prPosFilters.size > 0) {
    players = players.filter(p => prPosFilters.has(p.position) && (!p.is_rookie || isDrafted(p)));
  } else {
    players = players.filter(p => !p.is_rookie || isDrafted(p));
  }

  // Search filter - fuzzy match, sort by score when query present
  if (prSearchQuery.length > 0) {
    const scored = players
      .map(p => ({
        p,
        score: Math.max(prFuzzyScore(p.name, prSearchQuery), prFuzzyScore(p.search_name, prSearchQuery))
      }))
      .filter(x => x.score > 0)
      .sort((a, b) => b.score !== a.score ? b.score - a.score : prGetValue(b.p) - prGetValue(a.p));
    players = scored.map(x => x.p);
  } else {
    // Normal sort when no search query
    players.sort((a, b) => prSortCmp(a, b, sortBy));
    // Cap the board at the top 500 PLAYERS; picks are added back on top so the
    // list is always 500 players + N picks.
    const _picks = players.filter(p => p.position === 'PICK');
    const _nonPicks = players.filter(p => p.position !== 'PICK');
    if (_nonPicks.length > 500) _nonPicks.length = 500;
    players = _nonPicks.concat(_picks);
    players.sort((a, b) => prSortCmp(a, b, sortBy));
  }

  const list   = document.getElementById('prList');
  const empty  = document.getElementById('prEmpty');
  const count  = document.getElementById('prCount');
  const countRow = document.getElementById('prCountRow');
  const header = document.getElementById('prTableHeader');

  if (players.length === 0) {
    list.innerHTML = '';
    empty.style.display = 'block';
    header.style.display = 'none';
    if (countRow) countRow.style.display = 'none';
    prRenderPagination(0, 0);
    return;
  }

  const total = players.length;
  const totalPages = Math.max(1, Math.ceil(total / prPageSize));
  if (prPage > totalPages) prPage = totalPages;
  const start = (prPage - 1) * prPageSize;
  const end   = Math.min(start + prPageSize, total);
  const pageSlice = players.slice(start, end);
  
  // Store filtered players count for pagination navigation
  window.prFilteredPlayers = players;

  empty.style.display = 'none';
  header.style.display = 'grid';
  if (countRow) countRow.style.display = 'flex';
  count.textContent = 'Showing ' + (start + 1) + '–' + end + ' of ' + total + ' player' + (total !== 1 ? 's' : '');

  const _PR_TIER_COLORS = ['','#10b981','#22d3ee','#3b82f6','#8b5cf6','#a855f7','#f59e0b','#f97316','#94a3b8','#64748b'];
  const _PR_TIER_LABELS = ['','Elite','Star','High-End Starter','Starter','Flex','Bench','Deep Bench','Handcuff','Fringe'];

  list.innerHTML = '';
  let prevTier = null;
  pageSlice.forEach((p, i) => {
    const _tier = (sortBy === 'value' || sortBy === 'rank') ? prGetTier(p) : null;
    if (_tier && _tier !== prevTier) {
      const tc = _PR_TIER_COLORS[_tier] || '#64748b';
      const tl = _PR_TIER_LABELS[_tier] || ('Tier ' + _tier);
      const div = document.createElement('div');
      div.className = 'pr-tier-divider';
      div.innerHTML =
        `<div class="pr-tier-divider-line" style="background:${tc};"></div>` +
        `<span class="pr-tier-divider-label" style="color:${tc};" title="${tl}">T${_tier}</span>` +
        `<div class="pr-tier-divider-line" style="background:${tc};"></div>`;
      list.appendChild(div);
    }
    if (_tier) prevTier = _tier;
    const row = document.createElement('div');
    row.className = 'pr-player-row pr-grid-row';
    row.setAttribute('data-flip-key', 'p' + p.id);
    row.dataset.pid = String(p.id);
    // Draft picks aren't real players (no player/prospect modal to open), so the
    // row isn't clickable for them.
    const isPick = p.position === 'PICK';
    if (!isPick) {
      const _draftedWarm = p.is_rookie && p.team && p.team !== 'FA';
      if (!p.is_rookie || _draftedWarm) row.dataset.pmWarmId = String(p.id);
      else row.dataset.pmProspectOnly = '1';
      row.style.cursor = 'pointer';
      row.onclick = function(e) {
        e.stopPropagation();
        const _drafted = p.is_rookie && p.team && p.team !== 'FA';
        if (p.is_rookie && !_drafted) {
          if (typeof rkOpenModal === 'function') {
            rkOpenModal(p);
          } else {
            openProspectModal(p.id, p.name || 'Unknown');
          }
        } else {
          openPlayerModal(p.id, p.name || 'Unknown');
        }
      };
    }

    const _drafted = p.is_rookie && p.team && p.team !== 'FA';
    const displayRank = (p.position === 'PICK' || (p.is_rookie && !_drafted)) ? '' : (_rankMap.get(String(p.id)) ?? (start + i + 1));
    const posRank = prPosRankLabel(p);
    const age = p.age != null ? Number(p.age).toFixed(1) : '–';
    const val = prGetValue(p);

    // Watchlist star, immediately after the player name (before badges).
    const wlStar = (!isPick && typeof _isWatched === 'function' && _isWatched(p.id))
      ? '<span class="pr-wl-star" title="On your watchlist" aria-hidden="true">&#9733;</span>' : '';
    let badges = '';
    if (prIsRookie(p.id) || p.is_rookie) badges += '<span class="player-badge player-badge-rookie player-badge-collapsible" title="Rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> <span class="player-badge-label">ROOKIE</span></span>';
    else if (prIsProspect(p.id)) badges += '<span class="player-badge player-badge-prospect player-badge-collapsible" title="Prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> <span class="player-badge-label">PROSPECT</span></span>';
    if (prIsBreakout(p.id)) badges += '<span class="player-badge player-badge-breakout player-badge-collapsible" title="Breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> <span class="player-badge-label">BREAKOUT</span></span>';

    // Movement must match the ordering of the current view. We only track
    // dynasty value history, so:
    //   • Redraft has no redraft-ordered movement -- hide the arrow rather than
    //     show a mismatched dynasty delta on redraft-sorted rows.
    //   • Dynasty SF uses the SF-ordered delta (QBs move very differently in
    //     Superflex); Dynasty 1QB uses the 1QB delta.
    const rankChange = prScoringType === 'redraft'
      ? null
      : (prLeagueType === 'sf' ? p.sf_rank_change_7d : p.rank_change_7d);
    // Align the sparkline to the value actually displayed in this row. The
    // /api/sparklines series ends at the calibrated base value (no TE premium,
    // base league size), while `val` = prGetValue(p) applies the TE premium and
    // the size-specific column - so rescale the whole series so its last point
    // equals `val`. Preserves the 7-day shape; guarantees the endpoint matches.
    const sparkData = prAlignSpark(prGetSparkData(p.id), val);

    // Arrows column: the value-trend sparkline only. Rank movement now sits
    // beside the rank number, so there's no chevron fallback here.
    let arrowCell = '';
    if (sparkData && sparkData.length >= 2) {
      arrowCell = `<canvas class="pr-sparkline" data-pid="${p.id}" title="7-day value trend"></canvas>`;
    }

    // Subline: mini sparkline + pos rank + team (plain text, the live
    // "POSRANK · TEAM" join), then a muted middle-dot separator and the
    // age (from the payload bDay via calculateAgeFromBirthday; omitted
    // for a row whose payload carries neither).
    let sublineHTML = '';
    if (!isPick) {
      const _subParts = [posRank, (p.team && p.team !== '–') ? p.team : ''].filter(Boolean).join(' · ');
      const _subSpark = (sparkData && sparkData.length >= 2)
        ? `<canvas class="pr-spark-sub-canvas" data-pid="${p.id}" title="7-day value trend"></canvas>` : '';
      const _ageSeg = (p.age != null)
        ? ((_subParts ? '<span class="pr-age-sep" aria-hidden="true">·</span>' : '')
           + `<span class="pr-age-inline">Age ${Number(p.age).toFixed(1)}</span>`)
        : '';
      if (_subParts || _subSpark || _ageSeg) {
        sublineHTML = `<span class="pr-sub">${_subSpark}<span>${_subParts}</span>${_ageSeg}</span>`;
      }
    }

    // Overall rank movement, shown next to the rank number. Only meaningful when
    // the rank IS the overall value board (not the ADP / PPG / age / pos views).
    let rankDeltaHTML = '';
    const _rankIsOverall = (sortBy === 'value' || sortBy === 'rank');
    if (displayRank && _rankIsOverall && rankChange != null && rankChange !== 0) {
      const _up = rankChange > 0;
      const _n = Math.abs(rankChange);
      rankDeltaHTML = `<span class="pr-rank-delta ${_up ? 'up' : 'down'}" title="${_n} spot${_n !== 1 ? 's' : ''} overall in 7 days">${_up ? '▲' : '▼'}${_n}</span>`;
    }

    const sortDisplay = p.position === 'PICK' && sortBy === 'age' ? '–' : valueMeta.cell(p);
    let sortDisplayHTML;
    if (p.position !== 'PICK' && sortBy === 'total_pts' && p.total_pts != null && isMobile) {
      // Phone (shipped #2716 semantics): the sorted-by metric is the big
      // number -- Total Points on top, Value underneath as a labeled
      // sublabel. Desktop keeps both numbers in their own columns instead.
      sortDisplayHTML = `<span style="display:flex;flex-direction:column;align-items:center;line-height:1.2;">`
        + `<span>${Number(p.total_pts).toFixed(1)}</span>`
        + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">VAL ${sortDisplay}</span>`
        + `</span>`;
    } else {
      sortDisplayHTML = sortDisplay;
    }

    // Total Points column: always the player's total in its own column;
    // while sorting by Total Points its positional rank rides underneath.
    let totalCellHTML = '–';
    if (p.position !== 'PICK' && p.total_pts != null) {
      if (sortBy === 'total_pts' && p.total_pts_rank) {
        totalCellHTML = `<span style="display:flex;flex-direction:column;align-items:center;line-height:1.2;">`
          + `<span>${Number(p.total_pts).toFixed(1)}</span>`
          + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${p.position}${p.total_pts_rank}</span>`
          + `</span>`;
      } else {
        totalCellHTML = Number(p.total_pts).toFixed(1);
      }
    }

    // Dedicated PPG column: always visible next to Value so both read at a
    // glance. When sorting by PPG the cell also carries the positional PPG
    // rank as a sublabel (the detail that used to replace the Value column).
    let ppgCellHTML = '–';
    if (p.position !== 'PICK' && p.ppg != null) {
      if (sortBy === 'ppg') {
        const pRank = p.ppg_rank ? (p.position + p.ppg_rank) : '-';
        ppgCellHTML = `<span style="display:flex;flex-direction:column;align-items:center;line-height:1.2;">`
          + `<span>${Number(p.ppg).toFixed(1)}</span>`
          + `<span style="font-size:10px;font-weight:600;color:var(--text-muted);">${pRank}</span>`
          + `</span>`;
      } else {
        ppgCellHTML = Number(p.ppg).toFixed(1);
      }
    }

    if (adpView) {
      // ADP view: # | Player | (Pos | Age | Team on desktop) | one ADP cell per
      // source (active source highlighted).
      row.classList.add('pr-adp-mode');
      row.style.gridTemplateColumns = ADP_GRID;
      const metaCells = adpExtra
        ? ('<span class="pr-adp-meta">' + posRank + '</span>' +
           '<span class="pr-adp-meta">' + (p.position === 'PICK' ? '–' : age) + '</span>' +
           '<span class="pr-adp-meta">' + (p.team || '–') + '</span>')
        : '';
      // The ADP the row is sorted by (the highlighted source's), used as the
      // baseline the other sources' arrows compare against for this player.
      const _activeAdp = prAdpSourceVal(p, adpActive);
      row.innerHTML =
        '<span class="pr-rank pr-adp-pin pr-adp-pin-rank">'  + (displayRank ? '#' + displayRank : '–') + '</span>' +
        '<span class="pr-name pr-adp-pin pr-adp-pin-player' + (isPick ? '' : ' player-clickable') + '">'  + prHeadshotHTML(p, isPick) + '<span class="pr-name-text">' + (p.name || 'Unknown') + wlStar + badges + '</span></span>' +
        metaCells +
        adpCols.map(function (c) {
          const v = prAdpSourceVal(p, c.value);
          const on = c.value === adpActive;
          // Arrow vs the sorted source: a lower ADP (earlier pick) means this
          // source ranks the player HIGHER than the sorted one → green ▲; a
          // higher ADP (later pick) ranks him LOWER → red ▼. The sorted source
          // itself is the baseline, and equal/missing values get no arrow.
          let arrow = '';
          if (!on && v != null && _activeAdp != null && v !== _activeAdp) {
            const _higher = v < _activeAdp;
            arrow = '<span class="pr-adp-arrow ' + (_higher ? 'up' : 'down') +
              '" aria-hidden="true">' + (_higher ? '▲' : '▼') + '</span>';
          }
          // Wrap the number in its own box and hang the arrow off it (absolutely
          // positioned in CSS), so the arrow adds no layout width and the number
          // stays centered in the column whether or not an arrow is present.
          return '<span class="pr-adp-cell' + (on ? ' pr-adp-cell-active' : '') + '">' +
            '<span class="pr-adp-num">' + (v != null ? v.toFixed(1) : '–') + arrow + '</span></span>';
        }).join('');
    } else {
      row.innerHTML =
        '<span class="pr-rank">'  + (displayRank ? '#' + displayRank : '–') + rankDeltaHTML + '</span>' +
        '<span class="pr-name' + (isPick ? '' : ' player-clickable') + '">' + prHeadshotHTML(p, isPick) + '<span class="pr-name-text"><span class="pr-name-main">'  + (p.name || 'Unknown') + wlStar + badges + '</span>' + sublineHTML + '</span></span>' +
        '<span class="pr-ppg">' + ppgCellHTML + '</span>' +
        '<span class="pr-total">' + totalCellHTML + '</span>' +
        '<span class="pr-value">' + sortDisplayHTML + '</span>';

      if (sparkData && sparkData.length >= 2) {
        const cnvSub = row.querySelector('.pr-spark-sub-canvas');
        if (cnvSub) _prDrawSparkline(cnvSub, sparkData, _prSparkAnimate, 26, 14);
      }
    }

    list.appendChild(row);
  });

  prRenderPagination(prPage, totalPages);
}

function prRenderPagination(page, totalPages) {
  let bar = document.getElementById('prPagination');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'prPagination';
    bar.className = 'pagination';
    document.getElementById('prList').insertAdjacentElement('afterend', bar);
    bar.innerHTML =
      `<div class="pagination-info"><span id="prPaginationText"></span></div>` +
      `<div class="pagination-controls">` +
        `<button class="pagination-btn" id="prPrevBtn" onclick="prGoPage('prev')" disabled>` +
          `<i class="fa-solid fa-chevron-left"></i> Previous` +
        `</button>` +
        `<div class="pagination-pages" id="prPageNumbers"></div>` +
        `<button class="pagination-btn" id="prNextBtn" onclick="prGoPage('next')" disabled>` +
          `Next <i class="fa-solid fa-chevron-right"></i>` +
        `</button>` +
      `</div>`;
  }

  if (totalPages <= 1) {
    bar.style.display = 'none';
    return;
  }

  bar.style.display = 'flex';

  const start = (page - 1) * prPageSize + 1;
  const end = Math.min(page * prPageSize, window.prFilteredPlayers.length);
  const total = window.prFilteredPlayers.length;
  document.getElementById('prPaginationText').textContent = `Showing ${start}-${end} of ${total} players`;

  document.getElementById('prPrevBtn').disabled = page === 1;
  document.getElementById('prNextBtn').disabled = page === totalPages;

  // Page number buttons - show up to 5, centred on current page
  const pageNumbers = document.getElementById('prPageNumbers');
  pageNumbers.innerHTML = '';
  const maxPages = 5;
  let startPage = Math.max(1, page - Math.floor(maxPages / 2));
  let endPage = Math.min(totalPages, startPage + maxPages - 1);
  if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);
  for (let i = startPage; i <= endPage; i++) {
    const btn = document.createElement('button');
    btn.className = 'pagination-page' + (i === page ? ' active' : '');
    btn.textContent = i;
    btn.onclick = (function(n){ return function(){ prGoPage(n); }; })(i);
    pageNumbers.appendChild(btn);
  }
}

function prGoPage(p) {
  if (p === 'prev') {
    prPage = Math.max(1, prPage - 1);
  } else if (p === 'next') {
    const totalPages = Math.ceil(window.prFilteredPlayers.length / prPageSize);
    prPage = Math.min(totalPages, prPage + 1);
  } else {
    prPage = p;
  }
  prRender();
  // Scroll to top of player list
  const el = document.getElementById('prTableHeader');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Wire up search input
(function() {
  const inp   = document.getElementById('prSearch');
  const clear = document.getElementById('prSearchClear');
  if (!inp) return;

  // Honor WebSite SearchAction /players?q= so Google sitelinks and shared
  // search URLs actually filter the rankings table.
  try {
    const q = new URLSearchParams(window.location.search).get('q');
    if (q) {
      inp.value = q;
      prSearchQuery = q.trim();
      if (clear) clear.style.display = prSearchQuery.length > 0 ? 'block' : 'none';
    }
  } catch (e) {}

  inp.addEventListener('input', function() {
    prSearchQuery = inp.value.trim();
    clear.style.display = prSearchQuery.length > 0 ? 'block' : 'none';
    prPage = 1;
    prRender();
  });
})();

// Export current filtered list as CSV
function prExportCSV() {
  const players = window.prFilteredPlayers && window.prFilteredPlayers.length
    ? window.prFilteredPlayers : prAllPlayers;
  if (!players || !players.length) return;
  const q = (s) => '"' + String(s || '').replace(/"/g, '""') + '"';
  const header = ['Rank','Name','Position','Team','Age','Value','1QB Value','SF Value','7d Rank Change','PPG','Total Pts','PPG Season'];
  const rows = players.map((p, i) => {
    const val1qb = Number(p[prValueKey(false)] ?? p.value ?? 0);
    const valsf  = Number(p[prValueKey(true)]  ?? p.sf_value ?? 0);
    // PPG / Total Points exports carry the actual overall rank (the same
    // server rank the table shows), not the export row number.
    const _srvRank = prSortKey === 'ppg' ? p.ppg_ovr_rank
      : (prSortKey === 'total_pts' ? p.total_pts_ovr_rank : null);
    return [
      _srvRank != null ? _srvRank : i + 1,
      q(p.name),
      p.position || '',
      p.team || '',
      p.age != null ? Number(p.age).toFixed(1) : '',
      Number(prGetValue(p)).toFixed(1),
      val1qb.toFixed(1),
      valsf.toFixed(1),
      p.rank_change_7d != null ? p.rank_change_7d : '',
      p.ppg != null ? Number(p.ppg).toFixed(1) : '',
      p.total_pts != null ? Number(p.total_pts).toFixed(1) : '',
      p.ppg_season != null ? p.ppg_season : ''
    ];
  });
  const csv = [header, ...rows].map(r => r.join(',')).join('\\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'player_rankings.csv'; a.click();
  URL.revokeObjectURL(url);
  if (window.showToast) showToast('Player rankings downloaded!');
}

// Document-level listeners: bound once. This module re-executes on a mobile
// soft-nav (the dock swaps #page-root and re-runs the page script), so guard
// these so they don't stack a new listener on every navigation. They re-query
// their targets at event time, so a single binding stays correct across swaps.
if (!window.__prDocBound) {
  window.__prDocBound = true;

  // '/' focuses search from anywhere on the page
  document.addEventListener('keydown', function(e) {
    if (e.key === '/' && !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      const inp = document.getElementById('prSearch');
      if (inp) { inp.focus(); inp.select(); }
    }
  });

  // Close settings panel when clicking outside
  document.addEventListener('click', function(e) {
    const panel = document.getElementById('prSettingsPanel');
    const btn = document.getElementById('prSettingsBtn');

    if (panel && btn && panel.style.display === 'block') {
      if (!panel.contains(e.target) && !btn.contains(e.target)) {
        panel.style.display = 'none';
        btn.classList.remove('active');
      }
    }
    const lpanel = document.getElementById('prLegendPanel');
    const lbtn = document.getElementById('prLegendBtn');
    if (lpanel && lbtn && lpanel.style.display === 'block') {
      if (!lpanel.contains(e.target) && !lbtn.contains(e.target)) {
        lpanel.style.display = 'none';
        lbtn.classList.remove('active');
      }
    }
  });

  // Clickable sort headers (AM-table pattern): click a column to sort by it,
  // click again to flip direction. The ADP-source view swaps the header for
  // its own sortable columns, so clicks there are ignored. Bound once; the
  // header markup is rebuilt on soft-nav, so targets resolve at event time.
  function _prHeaderCellFromEvent(e) {
    var cell = e.target && e.target.closest ? e.target.closest('#prTableHeader [data-pr-sort-col]') : null;
    if (!cell) return null;
    var header = document.getElementById('prTableHeader');
    if (header && header.dataset.adpMode === '1') return null;
    return cell;
  }
  function _prHeaderSortKey(cell) {
    var col = cell.getAttribute('data-pr-sort-col');
    return col === 'player' ? 'name' : col === 'sort' ? prSortKey : col;
  }
  document.addEventListener('click', function(e) {
    var cell = _prHeaderCellFromEvent(e);
    if (cell) prHeaderSort(_prHeaderSortKey(cell));
  });
  document.addEventListener('keydown', function(e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    var cell = (e.target && e.target.closest && e.target.closest('#prTableHeader [data-pr-sort-col]'))
      ? _prHeaderCellFromEvent(e) : null;
    if (!cell) return;
    e.preventDefault();
    prHeaderSort(_prHeaderSortKey(cell));
  });
}

var prTierThresholds = {};

function prGetTier(p) {
  const lt  = prLeagueType || '1qb';
  const sz  = String(prLeagueSize || 10);
  const tbl = (prTierThresholds[lt] || {})[sz] || (prTierThresholds['1qb'] || {})['10'] || [];
  if (!tbl.length) return null;
  const val = prGetValue(p);
  for (let i = 0; i < tbl.length; i++) {
    if (val >= tbl[i]) return i + 1;
  }
  return tbl.length + 1; // catch-all - matches Python _ps_tier_of
}

// Load data
function prLoadData() {
  var gen = ++_prLoadGen;
  var loading = document.getElementById('prLoading');
  var list = document.getElementById('prList');
  // The /players route SSR-fills #prList and hides #prLoading. Unhiding the
  // skeleton here stacked it above the already-painted table until the API
  // returned. Keep the SSR rows on screen and only show a skeleton on a cold
  // load (empty list) or when retrying after an error.
  var hasSsrRows = !!(list && list.firstElementChild);
  var recovering = !!(loading && loading.querySelector('.empty-state'));
  if (loading) {
    if (hasSsrRows && !recovering) {
      loading.style.display = 'none';
      loading.setAttribute('aria-busy', 'false');
    } else {
      loading.style.display = '';
      loading.setAttribute('aria-busy', 'true');
      if (window.brLoadingState && recovering) {
        window.brLoadingState(loading, { spinner: true, message: 'Loading players…' });
      }
    }
  }
Promise.all([
  // Fail loud on a non-2xx: the server can return a 500 whose body is still
  // valid JSON (the _api_err shape), which would otherwise parse into an empty
  // players list and render a blank table with no error shown.
  // Pass league context so the server can add a Yahoo ADP column for Yahoo
  // leagues (Yahoo ADP needs a league id + token).
  (function () {
    let _u = '/api/league-players';
    const _q = [];
    if (window.__leagueId) _q.push('league_id=' + encodeURIComponent(window.__leagueId));
    if (window.__platform) _q.push('platform=' + encodeURIComponent(window.__platform));
    if (_q.length) _u += '?' + _q.join('&');
    // Default cache mode: the server sends ETag + max-age=60, so repeat
    // visits get 304s instead of full re-downloads. no-store is not used.
    return (typeof window.brFetchWithTimeout === 'function'
      ? window.brFetchWithTimeout(_u, {}, 30000)
      : fetch(_u)
    ).then(r => {
      if (!r.ok) throw new Error('league-players HTTP ' + r.status);
      return r.json();
    });
  })(),
  // Indicators deferred: loaded after first paint via requestIdleCallback.
  // Row chips render without them initially, then update when they arrive.
  Promise.resolve({})
]).then(([resp, indicators]) => {
  if (gen !== _prLoadGen) return;
  prIndicators = indicators || {};
  // Support both old (array) and new (object with players + tier_thresholds) format
  const rawPlayers = Array.isArray(resp) ? resp : (resp.players || []);
  prTierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};
  prAdpSourceOptions = (!Array.isArray(resp) && resp.adp_source_options) ? resp.adp_source_options : {};
  prAdpSources = (!Array.isArray(resp) && resp.adp_sources) ? resp.adp_sources : {};
  prAdpColumns = (!Array.isArray(resp) && Array.isArray(resp.adp_columns)) ? resp.adp_columns : [];

  // Helper function to calculate precise age from birthday
  function calculateAgeFromBirthday(bDay) {
    if (!bDay) return null;
    try {
      const parts = bDay.split('/');
      if (parts.length !== 3) return null;
      const [month, day, year] = parts.map(Number);
      const birthDate = new Date(year, month - 1, day);
      const today = new Date();
      
      // Match the server's age_from_bday exactly - the age the player
      // modal shows: total days / 365.25, floor-truncated to 1 decimal.
      // The old round-to-nearest here made the board read 31.1 where the
      // modal read 31.0 for the same birthday (Mahomes, 2026-10-22).
      const days = Math.floor((today - birthDate) / (1000 * 60 * 60 * 24));
      return Math.floor((days / 365.25) * 10) / 10;
    } catch (e) {
      return null;
    }
  }

  // Label the PPG column with the season the numbers actually describe,
  // so a stale-looking column can never pass as current by accident.
  const _ppgHead = document.getElementById('prPpgHeader');
  if (_ppgHead) {
    const _ppgSeason = (rawPlayers.find(p => p && p.ppg_season) || {}).ppg_season;
    const _ppgBasis = (!Array.isArray(resp) && resp.ppg_scoring === 'league') ? 'league scoring' : 'full PPR';
    _ppgHead.title = _ppgSeason ? (_ppgSeason + ' season PPG (' + _ppgBasis + ')') : ('Season PPG (' + _ppgBasis + ')');
  }

  prAllPlayers = rawPlayers
    .filter(p => p && p.id != null)
    .map(p => {
      // Calculate precise age from birthday if available
      const preciseAge = calculateAgeFromBirthday(p.bDay);
      
      return {
        id:               String(p.id),
        name:             p.name || p.full_name || 'Unknown',
        team:             p.team || '',
        position:         String(p.position || '').toUpperCase(),
        // The payload carries both (stamped from the players index server
        // side): bDay drives the precise age above, espnHeadshot drives the
        // row headshot. Both used to be dropped by this whitelist map.
        bDay:             p.bDay || null,
        espnHeadshot:     p.espnHeadshot || '',
        age:              preciseAge !== null ? preciseAge : (p.age != null ? Number(p.age) : null),
        value:            Number(p.value    || 0),
        value_8:          Number(p.value_8  || p.value    || 0),
        value_12:         Number(p.value_12 || p.value    || 0),
        value_14:         Number(p.value_14 || p.value    || 0),
        sf_value:         Number(p.sf_value    || p.value || 0),
        sf_value_8:       Number(p.sf_value_8  || p.sf_value || p.value || 0),
        sf_value_12:      Number(p.sf_value_12 || p.sf_value || p.value || 0),
        sf_value_14:      Number(p.sf_value_14 || p.sf_value || p.value || 0),
        redraft_value_1qb: p.redraft_value_1qb != null ? Number(p.redraft_value_1qb) : null,
        redraft_value_sf:  p.redraft_value_sf  != null ? Number(p.redraft_value_sf)  : null,
        redraft_value_8:   p.redraft_value_8   != null ? Number(p.redraft_value_8)   : null,
        redraft_value_12:  p.redraft_value_12  != null ? Number(p.redraft_value_12)  : null,
        redraft_value_14:  p.redraft_value_14  != null ? Number(p.redraft_value_14)  : null,
        redraft_sf_value_8:  p.redraft_sf_value_8  != null ? Number(p.redraft_sf_value_8)  : null,
        redraft_sf_value_12: p.redraft_sf_value_12 != null ? Number(p.redraft_sf_value_12) : null,
        redraft_sf_value_14: p.redraft_sf_value_14 != null ? Number(p.redraft_sf_value_14) : null,
        avg_pick:            p.avg_pick            != null ? Number(p.avg_pick)            : null,
        sf_avg_pick:         p.sf_avg_pick         != null ? Number(p.sf_avg_pick)         : null,
        redraft_avg_pick:    p.redraft_avg_pick    != null ? Number(p.redraft_avg_pick)    : null,
        sf_redraft_avg_pick: p.sf_redraft_avg_pick != null ? Number(p.sf_redraft_avg_pick) : null,
        rookie_avg_pick:     p.rookie_avg_pick     != null ? Number(p.rookie_avg_pick)     : null,
        sf_rookie_avg_pick:  p.sf_rookie_avg_pick  != null ? Number(p.sf_rookie_avg_pick)  : null,
        // Per-source ADP for the sort-by-ADP columns. This whitelist map drops
        // any field not named here, so it must be carried through explicitly or
        // every source cell renders as a dash.
        adp_by_source:       (p.adp_by_source && typeof p.adp_by_source === 'object') ? p.adp_by_source : null,
        sf_rank_change_7d:   p.sf_rank_change_7d != null ? Number(p.sf_rank_change_7d) : null,
        pos_rank_label:   p.pos_rank_label    || '',
        sf_pos_rank_label:p.sf_pos_rank_label || '',
        pos_rank:         Number(p.pos_rank    || 9999),
        sf_pos_rank:      Number(p.sf_pos_rank || 9999),
        search_name:      p.search_name || '',
        is_rookie:        p.is_rookie === true,
        rank_change_7d:   p.rank_change_7d != null ? Number(p.rank_change_7d) : null,
        ppg:              p.ppg != null ? Number(p.ppg) : null,
        total_pts:        p.total_pts != null ? Number(p.total_pts) : null,
        ppg_raw:          p.ppg_raw != null ? Number(p.ppg_raw) : null,
        total_pts_raw:    p.total_pts_raw != null ? Number(p.total_pts_raw) : null,
        ppg_rank:         p.ppg_rank != null ? Number(p.ppg_rank) : null,
        total_pts_rank:   p.total_pts_rank != null ? Number(p.total_pts_rank) : null,
        ppg_ovr_rank:     p.ppg_ovr_rank != null ? Number(p.ppg_ovr_rank) : null,
        total_pts_ovr_rank: p.total_pts_ovr_rank != null ? Number(p.total_pts_ovr_rank) : null,
        ppg_season:       p.ppg_season || null,
      };
    })
    .filter(p => ['QB','RB','WR','TE','PICK'].includes(p.position) || p.is_rookie)

  // A 200 response with an empty/malformed payload (e.g. the value table failed
  // to load server-side) would otherwise render as a silent blank table.
  if (!prAllPlayers.length) {
    throw new Error('league-players returned no players');
  }

  var _prLoadEl = document.getElementById('prLoading');
  if (_prLoadEl) {
    _prLoadEl.style.display = 'none';
    _prLoadEl.setAttribute('aria-busy', 'false');
  }
  prLoaded = true;
  prRender();
  // Lazy-load sparklines - re-render with sparkline data once ready
  fetch('/api/sparklines?v=4').then(r => r.json()).then(function(data) {
    prSparklines = data || {};
    _prSparkAnimate = true;   // sweep the lines on for this first reveal only
    prRender();
    _prSparkAnimate = false;
  }).catch(function() {});
  // Lazy-load indicators - row chips (rookie/breakout/elite/prospect) update
  // once they arrive. Deferred out of the critical path via idle callback.
  var _loadIndicators = function() {
    fetch('/api/player-indicators?league_type=1qb&league_size=10')
      .then(function(r) { return r.json(); }).then(function(d) {
        if (gen !== _prLoadGen) return;
        prIndicators = d || {};
        prRender();
      }).catch(function() {});
  };
  if ('requestIdleCallback' in window) {
    requestIdleCallback(_loadIndicators, { timeout: 3000 });
  } else {
    setTimeout(_loadIndicators, 100);
  }
}).catch(err => {
  if (gen !== _prLoadGen) return;
  // A reload/navigation aborts the in-flight hydrate. Don't replace the SSR
  // table with a full-page error for a request that is no longer current.
  if (err && err.name === 'AbortError') return;
  console.error('Error loading player rankings:', err);
  var listEl = document.getElementById('prList');
  if (listEl && listEl.firstElementChild) return;
  var el = document.getElementById('prLoading');
  if (!el) return;
  el.style.display = '';
  el.removeAttribute('aria-hidden');
  el.setAttribute('aria-busy', 'false');
  if (window.brErrorState) {
    window.brErrorState(el, 'Failed to load players. Check your connection and try again.', prLoadData, { compact: true });
  } else {
    el.innerHTML = '<div class="empty-state empty-state-error">Failed to load players. Please refresh.</div>';
  }
});
}
// ── Hover quick-look card (desktop pointers only) ─────────────────────
// After a 600ms hover intent on a row, a horizontal strip opens under (or
// above) the row with what the row does not already show: both format
// values, the 30-day value trend, the last 3 game logs, and the next
// matchup in Schedule Rankings terms. Data the page cannot already know
// comes from /api/player-quicklook (trend + next matchup) and
// /api/player-game-logs (the player modal's own endpoint). Phones get
// nothing here: tap already opens the full player modal in one tap, and
// the card must never become a toll booth in front of it.
var PR_QL_DELAY = 600;
var prQlEl = null;
var prQlPid = null;
var prQlRow = null;
var prQlShowTimer = null;
var prQlHideTimer = null;
var prQlCache = {};
var prQlRankCache = { sig: '', by: {} };
var prQlCanHover = !!(window.matchMedia
  && window.matchMedia('(hover: hover) and (pointer: fine)').matches);

function prQlEsc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// prGetValue for a forced format (the card shows 1QB and SF side by side,
// whatever the board is currently set to).
function prQlValueFor(p, isSf) {
  var base;
  if (prScoringType === 'redraft') {
    base = isSf
      ? Number(p.redraft_value_sf != null ? p.redraft_value_sf : (p.redraft_value_1qb || 0))
      : Number(p.redraft_value_1qb || 0);
  } else if (isSf) {
    base = Number(p[prValueKey(true)] != null ? p[prValueKey(true)] : (p.sf_value != null ? p.sf_value : (p.value || 0)));
  } else {
    base = Number(p[prValueKey(false)] != null ? p[prValueKey(false)] : (p.value || 0));
  }
  base *= prTeBoost(p.position);
  return Math.round(base * 10) / 10;
}

// Overall + positional ranks for one format, computed over the loaded
// board (players only, value > 0) and cached per settings signature.
function prQlRanksFor(isSf) {
  var sig = [prScoringType, prLeagueSize, prTePremium].join('|');
  if (prQlRankCache.sig !== sig) prQlRankCache = { sig: sig, by: {} };
  var key = isSf ? 'sf' : 'qb';
  if (prQlRankCache.by[key]) return prQlRankCache.by[key];
  var rows = prAllPlayers.filter(function(p) {
    return p.position !== 'PICK' && prQlValueFor(p, isSf) > 0;
  });
  rows.sort(function(a, b) { return prQlValueFor(b, isSf) - prQlValueFor(a, isSf); });
  var ovr = {}, posCount = {}, pos = {};
  rows.forEach(function(p, i) {
    ovr[String(p.id)] = i + 1;
    var pc = (posCount[p.position] = (posCount[p.position] || 0) + 1);
    pos[String(p.id)] = (p.position || '') + pc;
  });
  prQlRankCache.by[key] = { ovr: ovr, pos: pos };
  return prQlRankCache.by[key];
}

// Long names stay whole: each part is unbreakable, a suffix joins its
// surname, and the size steps down as the name lengthens (ql-name-lg/xl).
function prQlNameHTML(name) {
  var parts = String(name || '').split(/\s+/).filter(Boolean);
  var SUF = ['Jr.', 'Sr.', 'II', 'III', 'IV', 'V'];
  if (parts.length >= 3 && SUF.indexOf(parts[parts.length - 1]) !== -1) {
    parts = parts.slice(0, -2)
      .concat([parts[parts.length - 2] + ' ' + parts[parts.length - 1]]);
  }
  var len = parts.join(' ').length;
  var cls = len > 19 ? ' ql-name-xl' : len > 15 ? ' ql-name-lg' : '';
  return '<div class="ql-name' + cls + '">' + parts.map(function(w) {
    return '<span class="ql-nw">' + prQlEsc(w) + '</span>';
  }).join(' ') + '</div>';
}

// Same headshot chain as prHeadshotHTML (ESPN combiner, Sleeper CDN
// fallback, initials last), at card size: combiner width 360 for the
// 168px-wide identity panel.
function prQlHeadshotHTML(p) {
  if (p.position === 'PICK') {
    return '<span class="ql-photo ql-photo-pick"><i class="fa-solid fa-ticket"></i></span>';
  }
  var name = String(p.name || '');
  var toks = name.split(/\s+/).filter(Boolean);
  var initials = toks.length
    ? (toks[0].charAt(0) + (toks.length > 1 ? toks[toks.length - 1].charAt(0) : '')).toUpperCase()
    : '?';
  var _style = ' style="background:var(--accent-soft);color:var(--accent)"';
  var mono = '<span class="ql-mono" style="display:none" aria-hidden="true">' + initials + '</span>';
  var espn = String(p.espnHeadshot || '');
  var sleeper = p.id ? ('https://sleepercdn.com/content/nfl/players/thumb/' + encodeURIComponent(p.id) + '.jpg') : '';
  if (!espn && !sleeper) {
    return '<span class="ql-photo"><span class="ql-mono ql-mono-on"' + _style + '>' + initials + '</span></span>';
  }
  var hi = espn ? prHiResHeadshot(espn, 360) : '';
  var src = hi || sleeper;
  var fb = [];
  if (hi && espn && hi !== espn) fb.push(espn);
  if (sleeper && sleeper !== src) fb.push(sleeper);
  return '<span class="ql-photo"><img class="ql-hs" src="' + src + '" data-fb="' + fb.join('|')
    + '" onerror="prImgFallback(this)" alt="" loading="lazy">' + mono + '</span>';
}

function prQlGameLine(p, g) {
  var s = (g && g.stats) || {};
  function n(k) { var v = s[k]; return v != null ? Number(v) : 0; }
  var bits = [];
  if (p.position === 'QB') {
    bits.push(n('pass_att') + ' att · ' + n('pass_yd') + ' yds · ' + n('pass_td') + ' TD');
    if (n('rush_att') >= 3) {
      bits.push(n('rush_att') + ' car · ' + n('rush_yd') + ' yds'
        + (n('rush_td') ? ' · ' + n('rush_td') + ' TD' : ''));
    }
  } else if (p.position === 'RB') {
    bits.push(n('rush_att') + ' car · ' + n('rush_yd') + ' yds · ' + n('rush_td') + ' TD');
    if (n('rec_tgt') > 0) {
      bits.push(n('rec_tgt') + ' tgt · ' + n('rec') + ' rec · ' + n('rec_yd') + ' yds'
        + (n('rec_td') ? ' · ' + n('rec_td') + ' TD' : ''));
    }
  } else if (p.position === 'WR' || p.position === 'TE') {
    bits.push(n('rec_tgt') + ' tgt · ' + n('rec') + ' rec · ' + n('rec_yd') + ' yds · ' + n('rec_td') + ' TD');
    if (n('rush_att') > 0) {
      bits.push(n('rush_att') + ' car · ' + n('rush_yd') + ' yds'
        + (n('rush_td') ? ' · ' + n('rush_td') + ' TD' : ''));
    }
  }
  return bits.join('<br>');
}

function prQlGamesHTML(p, payload) {
  var byYear = (payload && payload.game_logs_by_year) || {};
  var years = Object.keys(byYear).sort().reverse();
  var logs = null;
  for (var i = 0; i < years.length; i++) {
    if (Array.isArray(byYear[years[i]]) && byYear[years[i]].length) {
      logs = byYear[years[i]];
      break;
    }
  }
  if (!logs) return '<div class="ql-empty">No games yet this season</div>';
  var real = logs.filter(function(g) {
    if (!g || g.is_bye || g.is_projection || g.fantasy_pts == null) return false;
    var s = g.stats || {};
    return s.pass_yd != null || s.rush_att != null || s.rec != null || s.rec_tgt != null;
  }).slice().sort(function(a, b) { return (Number(a.week) || 0) - (Number(b.week) || 0); });
  var last = real.slice(-3).reverse();
  if (!last.length) return '<div class="ql-empty">No games yet this season</div>';
  return last.map(function(g) {
    var oppRaw = String(g.opponent || '').trim();
    var opp = !oppRaw || oppRaw === 'BYE' ? ''
      : oppRaw.charAt(0) === '@' ? '@ ' + oppRaw.slice(1) : 'vs ' + oppRaw;
    return '<div class="ql-game"><div class="ql-game-top"><span class="ql-game-wk">Wk '
      + (g.week != null ? g.week : '') + (opp ? ' · ' + prQlEsc(opp) : '')
      + '</span><span class="ql-game-pts">' + Number(g.fantasy_pts || 0).toFixed(1)
      + '</span></div><div class="ql-game-line">' + prQlGameLine(p, g) + '</div></div>';
  }).join('');
}

function prQlTrendSVG(vals) {
  var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
  var span = (max - min) || 1;
  var W = 200, H = 64, PAD = 5;
  var pts = vals.map(function(v, i) {
    var x = PAD + (W - 2 * PAD) * (vals.length === 1 ? 0 : i / (vals.length - 1));
    var y = H - PAD - (H - 2 * PAD) * ((v - min) / span);
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
  var lastPt = pts.split(' ').pop().split(',');
  var up = vals[vals.length - 1] >= vals[0];
  var color = up ? '#22c55e' : '#ef4444';
  return '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" aria-hidden="true">'
    + '<polyline points="' + pts + '" fill="none" stroke="' + color
    + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    + '<circle cx="' + lastPt[0] + '" cy="' + lastPt[1] + '" r="3.2" fill="' + color
    + '" stroke="var(--card-bg)" stroke-width="1.6"/></svg>';
}

var PR_QL_TIERS = {
  'Elite': ['#22c55e', 'rgba(34,197,94,0.16)'],
  'Good': ['#84cc16', 'rgba(132,204,22,0.16)'],
  'Tough': ['#f59e0b', 'rgba(245,158,11,0.16)'],
  'Brutal': ['#ef4444', 'rgba(239,68,68,0.16)']
};

function prQlNextHTML(p, nxt) {
  if (!nxt) return '<div class="ql-empty">No upcoming game scheduled</div>';
  var badge = (typeof window.brTeamBadge === 'function' && window.brTeamBadge(nxt.opp, 'md'))
    || '<span class="team-badge team-badge-md" style="--badge-primary:#334155;--badge-secondary:#64748b">'
      + prQlEsc(nxt.opp) + '</span>';
  var out = '<div class="ql-next-head">' + badge + '<div class="ql-next-id">'
    + '<div class="ql-next-vs">' + (nxt.home ? 'vs ' : '@ ') + prQlEsc(nxt.opp) + '</div>'
    + '<div class="ql-next-when">Wk ' + nxt.week
    + (nxt.when ? ' · ' + prQlEsc(nxt.when) : '') + '</div></div></div>';
  if (nxt.bye_week) {
    out += '<div class="ql-next-bye">Bye Wk ' + nxt.bye_week + '</div>';
  }
  if (nxt.def_ppg != null) {
    out += '<div class="ql-next-stat"><span class="ql-next-num">'
      + Number(nxt.def_ppg).toFixed(1)
      + '</span><span class="ql-next-cap">PPG allowed<br>to ' + prQlEsc(p.position)
      + 's</span></div>';
  }
  if (nxt.rank) {
    out += '<div class="ql-next-rank">#' + nxt.rank + ' easiest vs '
      + prQlEsc(p.position) + 's</div>';
  }
  if (nxt.tier && PR_QL_TIERS[nxt.tier]) {
    var tc = PR_QL_TIERS[nxt.tier];
    out += '<div class="ql-next-pill" style="color:' + tc[0] + ';background:' + tc[1]
      + '">' + nxt.tier + ' matchup</div>';
  }
  return out;
}

function prQlKey(pid) {
  return [pid, prLeagueType, prLeagueSize, prScoringType].join('|');
}

// Zones fill independently as their source lands: the quicklook fetch
// (trend + next) must never wait on the game-logs fetch, which can be
// slow on a cold cache.
function prQlFillQuick(p, q) {
  if (!prQlEl || prQlPid !== String(p.id)) return;
  var trendEl = prQlEl.querySelector('[data-ql-zone="trend"]');
  var nextEl = prQlEl.querySelector('[data-ql-zone="next"]');
  q = q || null;
  if (trendEl) {
    var hist = (q && q.history) || [];
    if (hist.length >= 2) {
      var pct = q.pct_7d;
      var pctHTML = pct == null ? '<span class="ql-trend-flat">7d change n/a</span>'
        : '<span class="' + (pct >= 0 ? 'ql-up' : 'ql-down') + '">'
          + (pct >= 0 ? '+' : '') + Number(pct).toFixed(1) + '% in 7d</span>';
      trendEl.innerHTML = '<div class="ql-zone-title">Value trend · 30d</div>'
        + '<div class="ql-chart">' + prQlTrendSVG(hist.map(function(h) { return h.v; }))
        + '</div><div class="ql-trend-foot"><span>' + prQlEsc(hist[0].d.slice(5))
        + '</span>' + pctHTML + '<span>Today</span></div>';
    } else {
      trendEl.innerHTML = '<div class="ql-zone-title">Value trend · 30d</div>'
        + '<div class="ql-empty">Not enough history yet</div>';
    }
  }
  if (nextEl) {
    nextEl.innerHTML = '<div class="ql-zone-title">Next matchup</div>'
      + '<div class="ql-next-panel">' + prQlNextHTML(p, q && q.next) + '</div>';
  }
  prQlPosition();
}

function prQlFillGames(p, payload) {
  if (!prQlEl || prQlPid !== String(p.id)) return;
  var gamesEl = prQlEl.querySelector('[data-ql-zone="games"]');
  if (gamesEl) {
    gamesEl.innerHTML = '<div class="ql-zone-title">Last 3 games</div>'
      + prQlGamesHTML(p, payload);
  }
  prQlPosition();
}

function prQlFetch(p) {
  var key = prQlKey(p.id);
  var entry = prQlCache[key] || (prQlCache[key] = {});
  if (entry.quickDone) {
    prQlFillQuick(p, entry.quick);
  } else if (!entry.quickBusy) {
    entry.quickBusy = true;
    var qs = '?league_type=' + encodeURIComponent(prLeagueType)
      + '&league_size=' + encodeURIComponent(prLeagueSize)
      + '&scoring_type=' + encodeURIComponent(prScoringType);
    fetch('/api/player-quicklook/' + encodeURIComponent(p.id) + qs)
      .then(function(r) { return r.ok ? r.json() : null; })
      .catch(function() { return null; })
      .then(function(q) {
        entry.quick = q;
        entry.quickDone = true;
        entry.quickBusy = false;
        prQlFillQuick(p, q);
      });
  }
  if (p.position === 'PICK') return;
  if (entry.gamesDone) {
    prQlFillGames(p, entry.games);
  } else if (!entry.gamesBusy) {
    entry.gamesBusy = true;
    fetch('/api/player-game-logs/' + encodeURIComponent(p.id))
      .then(function(r) { return r.ok ? r.json() : null; })
      .catch(function() { return null; })
      .then(function(g) {
        entry.games = g;
        entry.gamesDone = true;
        entry.gamesBusy = false;
        prQlFillGames(p, g);
      });
  }
}

function prQlBuild(p) {
  var isPick = p.position === 'PICK';
  var qbR = prQlRanksFor(false), sfR = prQlRanksFor(true);
  var vQb = prQlValueFor(p, false), vSf = prQlValueFor(p, true);
  var rankChange = prScoringType === 'redraft' ? null
    : (prLeagueType === 'sf' ? p.sf_rank_change_7d : p.rank_change_7d);
  var moveHTML;
  if (rankChange == null) {
    moveHTML = '<span class="ql-move-flat">Not tracked in redraft</span>';
  } else if (rankChange === 0) {
    moveHTML = '<span class="ql-move-flat">No move</span>';
  } else if (rankChange > 0) {
    moveHTML = '<span class="ql-up">▲ ' + rankChange + ' spots</span>';
  } else {
    moveHTML = '<span class="ql-down">▼ ' + Math.abs(rankChange) + ' spots</span>';
  }
  var meta = [p.team && p.team !== '–' ? p.team : '',
    p.age != null ? 'Age ' + Number(p.age).toFixed(1) : ''].filter(Boolean).join(' · ');
  var html = '<div class="ql-col ql-col-id">' + prQlHeadshotHTML(p) + prQlNameHTML(p.name)
    + '<div class="ql-meta"><span class="ql-pospill">' + prQlEsc(prPosRankLabel(p))
    + '</span>' + (meta ? '<span>' + prQlEsc(meta) + '</span>' : '') + '</div>'
    + '<div class="ql-move"><span class="ql-move-label">7d rank move</span>' + moveHTML
    + '</div></div>'
    + '<div class="ql-col ql-col-val"><div class="ql-zone-title">Value</div>'
    + '<div class="ql-values">'
    + '<div class="ql-vbox"><span class="ql-vlab">1QB value</span><span class="ql-vnum">'
    + prFormatValue(vQb) + '</span><span class="ql-vsub">'
    + (qbR.ovr[String(p.id)] ? 'Overall #' + qbR.ovr[String(p.id)] + ' · ' : '')
    + prQlEsc(qbR.pos[String(p.id)] || p.position) + '</span></div>'
    + '<div class="ql-vbox"><span class="ql-vlab">SF value</span><span class="ql-vnum">'
    + prFormatValue(vSf) + '</span><span class="ql-vsub">'
    + prQlEsc(sfR.pos[String(p.id)] || p.position) + ' in Superflex</span></div>'
    + '</div></div>'
    + '<div class="ql-col ql-col-trend" data-ql-zone="trend">'
    + '<div class="ql-zone-title">Value trend · 30d</div><div class="ql-skel"></div></div>';
  if (!isPick) {
    html += '<div class="ql-col ql-col-games" data-ql-zone="games">'
      + '<div class="ql-zone-title">Last 3 games</div><div class="ql-skel"></div></div>'
      + '<div class="ql-col ql-col-next" data-ql-zone="next">'
      + '<div class="ql-zone-title">Next matchup</div><div class="ql-skel"></div></div>';
  }
  return html;
}

function prQlPosition() {
  if (!prQlEl || !prQlRow) return;
  var r = prQlRow.getBoundingClientRect();
  var cw = prQlEl.offsetWidth, ch = prQlEl.offsetHeight;
  var left = Math.min(Math.max(12, r.left + r.width / 2 - cw / 2),
    window.innerWidth - cw - 12);
  var below = r.bottom + 10;
  var above = r.top - ch - 10;
  var placeAbove = below + ch > window.innerHeight - 8 && above >= 8;
  prQlEl.style.left = left + 'px';
  prQlEl.style.top = (placeAbove ? above : below) + 'px';
  prQlEl.classList.toggle('ql-above', placeAbove);
  var caret = Math.min(Math.max(26, r.left + 120 - left), cw - 26);
  prQlEl.style.setProperty('--ql-caret-x', caret + 'px');
}

function prQlShow(p, row) {
  prQlCancelHide();
  if (!prQlEl) {
    prQlEl = document.createElement('div');
    prQlEl.id = 'prQuicklook';
    prQlEl.className = 'ql-card';
    prQlEl.addEventListener('pointerenter', prQlCancelHide);
    prQlEl.addEventListener('pointerleave', function() { prQlScheduleHide(180); });
    document.body.appendChild(prQlEl);
  }
  if (prQlRow && prQlRow !== row) prQlRow.classList.remove('ql-on');
  prQlPid = String(p.id);
  prQlRow = row;
  row.classList.add('ql-on');
  prQlEl.innerHTML = prQlBuild(p);
  prQlEl.classList.toggle('ql-solo', p.position === 'PICK');
  prQlEl.style.visibility = 'hidden';
  prQlEl.classList.add('ql-open');
  prQlPosition();
  prQlEl.style.visibility = '';
  prQlFetch(p);
}

function prQlCancelShow() {
  if (prQlShowTimer) { clearTimeout(prQlShowTimer); prQlShowTimer = null; }
}
function prQlCancelHide() {
  if (prQlHideTimer) { clearTimeout(prQlHideTimer); prQlHideTimer = null; }
}
function prQlScheduleHide(ms) {
  prQlCancelHide();
  prQlHideTimer = setTimeout(prQlHide, ms);
}
function prQlHide() {
  prQlCancelShow();
  prQlCancelHide();
  if (prQlRow) prQlRow.classList.remove('ql-on');
  prQlRow = null;
  prQlPid = null;
  if (prQlEl) prQlEl.classList.remove('ql-open');
}

if (!window.__prQlBound) {
  window.__prQlBound = true;
  document.addEventListener('pointerover', function(e) {
    if (!prQlCanHover) return;
    var row = e.target && e.target.closest
      ? e.target.closest('#prList .pr-player-row') : null;
    if (!row || !row.dataset.pid) return;
    if (e.relatedTarget && row.contains(e.relatedTarget)) return;
    prQlCancelHide();
    var pid = row.dataset.pid;
    if (prQlPid === pid) return;
    var p = prAllPlayers.find(function(x) { return String(x.id) === pid; });
    if (!p) return;
    prQlCancelShow();
    prQlShowTimer = setTimeout(function() { prQlShow(p, row); }, PR_QL_DELAY);
  });
  document.addEventListener('pointerout', function(e) {
    if (!prQlCanHover) return;
    var row = e.target && e.target.closest
      ? e.target.closest('#prList .pr-player-row') : null;
    if (!row) return;
    if (e.relatedTarget && row.contains(e.relatedTarget)) return;
    if (e.relatedTarget && prQlEl && prQlEl.contains(e.relatedTarget)) {
      prQlCancelHide();
      return;
    }
    if (prQlPid === row.dataset.pid) prQlScheduleHide(220);
    else prQlCancelShow();
  });
  document.addEventListener('click', function(e) {
    if (e.target && e.target.closest && e.target.closest('#prList .pr-player-row')) {
      prQlHide();
    }
  });
  window.addEventListener('scroll', function() {
    if (prQlPid || prQlShowTimer) prQlHide();
  }, true);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') prQlHide();
  });
}

if (prScoringType === 'redraft') {
  try { prSetScoringType('redraft'); } catch (e) {}
}
prSyncHeaderSegs();
prSyncSortToggle();
if (!window.__prSortThumbBound) {
  window.__prSortThumbBound = true;
  window.addEventListener('resize', prSortThumb);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(prSortThumb);
}
prLoadData();
