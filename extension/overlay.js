(function () {
  const CLOCK_START = 75;
  const SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"];
  const POS = { QB: "#3b82f6", RB: "#22c55e", WR: "#f59e0b", TE: "#8b5cf6", FLEX: "#14b8a6", BN: "#64748b" };
  const SORTS = ["rec", "adp", "ps", "proj"];
  const SORT_LBL = { rec: "Rec", adp: "ADP", ps: "Pick Score", proj: "Proj" };
  const TEAM_NAMES = [
    "", "Midnight Express", "Gridiron Ghosts", "Capital Thunder", "Bayou Bandits",
    "Pacific Storm", "Ironclad FC", "You", "Desert Foxes", "Harbor Hawks",
    "Summit Wolves", "River City Aces", "Neon Nomads"
  ];
  const PLATFORMS = {
    sleeper: {
      url: "sleeper.app/draft/nfl/midnight-society",
      league: "Midnight Society",
      logo: "S",
      sync: "Sleeper · SYNCED"
    },
    yahoo: {
      url: "football.fantasysports.yahoo.com/f1/2026/draft",
      league: "The Commish Club",
      logo: "Y!",
      sync: "Yahoo · SYNCED"
    },
    espn: {
      url: "fantasy.espn.com/football/draft?leagueId=847291",
      league: "East Division Live",
      logo: "E",
      sync: "ESPN · SYNCED"
    }
  };

  const RAW = [
    ["Ja'Marr Chase","WR","CIN",10,1.2,1,20.4,25],
    ["Justin Jefferson","WR","MIN",6,1.8,1,19.9,26],
    ["Bijan Robinson","RB","ATL",5,2.4,1,19.6,23],
    ["CeeDee Lamb","WR","DAL",7,3.1,1,19.1,26],
    ["Jahmyr Gibbs","RB","DET",8,3.7,1,18.8,23],
    ["Saquon Barkley","RB","PHI",9,4.5,1,18.4,28],
    ["Amon-Ra St. Brown","WR","DET",8,5.2,1,18.1,25],
    ["Nico Collins","WR","HOU",14,6.1,1,17.6,26],
    ["Puka Nacua","WR","LAR",8,6.8,1,17.4,24],
    ["Malik Nabers","WR","NYG",14,7.4,1,17.1,22],
    ["Christian McCaffrey","RB","SF",9,8.2,1,17.8,29],
    ["De'Von Achane","RB","MIA",12,8.9,1,17.0,23],
    ["Brian Thomas Jr.","WR","JAC",8,9.6,1,16.7,22],
    ["A.J. Brown","WR","PHI",9,10.4,1,16.5,28],
    ["Drake London","WR","ATL",5,11.1,2,16.2,24],
    ["Josh Allen","QB","BUF",7,11.8,1,24.8,29],
    ["Lamar Jackson","QB","BAL",7,12.6,1,24.1,28],
    ["Derrick Henry","RB","BAL",7,13.2,2,16.0,31],
    ["Breece Hall","RB","NYJ",12,14.0,2,15.8,24],
    ["Kyren Williams","RB","LAR",8,14.8,2,15.6,25],
    ["Jonathan Taylor","RB","IND",14,15.5,2,15.4,26],
    ["Tee Higgins","WR","CIN",10,16.3,2,15.3,26],
    ["Ladd McConkey","WR","LAC",12,17.1,2,15.1,23],
    ["Josh Jacobs","RB","GB",10,17.9,2,14.9,27],
    ["Jaxon Smith-Njigba","WR","SEA",8,18.6,2,14.8,23],
    ["Brock Bowers","TE","LV",8,19.4,1,14.6,22],
    ["Tyreek Hill","WR","MIA",12,20.2,2,14.5,31],
    ["Ashton Jeanty","RB","LV",8,21.0,2,14.4,21],
    ["Jalen Hurts","QB","PHI",9,21.8,1,22.6,27],
    ["Bucky Irving","RB","TB",9,22.5,2,14.2,22],
    ["DK Metcalf","WR","PIT",9,23.3,2,14.0,27],
    ["Terry McLaurin","WR","WAS",12,24.1,2,13.9,29],
    ["Trey McBride","TE","ARI",8,24.8,1,13.8,25],
    ["Chase Brown","RB","CIN",10,25.6,2,13.7,25],
    ["Mike Evans","WR","TB",9,26.4,2,13.6,32],
    ["Marvin Harrison Jr.","WR","ARI",8,27.2,2,13.5,23],
    ["Joe Burrow","QB","CIN",10,28.0,1,21.8,28],
    ["Garrett Wilson","WR","NYJ",12,28.8,2,13.3,25],
    ["James Cook","RB","BUF",7,29.5,2,13.2,25],
    ["Jayden Daniels","QB","WAS",12,30.3,1,21.4,24],
    ["George Kittle","TE","SF",9,31.1,1,13.0,31],
    ["Courtland Sutton","WR","DEN",14,31.9,3,12.9,29],
    ["Alvin Kamara","RB","NO",12,32.7,3,12.8,30],
    ["Kenneth Walker","RB","SEA",8,33.5,3,12.7,25],
    ["Zay Flowers","WR","BAL",7,34.2,3,12.6,24],
    ["DJ Moore","WR","CHI",7,35.0,3,12.5,28],
    ["Sam LaPorta","TE","DET",8,35.8,2,12.4,24],
    ["Chuba Hubbard","RB","CAR",14,36.6,3,12.3,26],
    ["Chris Olave","WR","NO",12,37.4,3,12.2,25],
    ["Tetairoa McMillan","WR","CAR",14,38.1,3,12.1,22],
    ["Patrick Mahomes","QB","KC",10,38.9,2,20.6,30],
    ["James Conner","RB","ARI",8,39.7,3,12.0,30],
    ["George Pickens","WR","DAL",7,40.5,3,11.9,24],
    ["Rome Odunze","WR","CHI",7,41.3,3,11.8,23],
    ["David Montgomery","RB","DET",8,42.0,3,11.7,28],
    ["Xavier Worthy","WR","KC",10,42.8,3,11.6,22],
    ["DeVonta Smith","WR","PHI",9,43.6,3,11.5,26],
    ["Davante Adams","WR","LAR",8,44.4,3,11.4,32],
    ["Omarion Hampton","RB","LAC",12,45.2,3,11.3,22],
    ["Travis Kelce","TE","KC",10,46.0,2,11.2,36],
    ["Jameson Williams","WR","DET",8,46.8,3,11.1,24],
    ["Dak Prescott","QB","DAL",7,47.5,2,19.8,32],
    ["Jayden Reed","WR","GB",10,48.3,3,11.0,25],
    ["Isiah Pacheco","RB","KC",10,49.1,3,10.9,26],
    ["Khalil Shakir","WR","BUF",7,49.9,3,10.8,25],
    ["Mark Andrews","TE","BAL",7,50.7,2,10.7,30],
    ["TreVeyon Henderson","RB","NE",14,51.5,3,10.6,22],
    ["Calvin Ridley","WR","TEN",10,52.3,4,10.5,30],
    ["Justin Herbert","QB","LAC",12,53.0,2,19.2,27],
    ["Jordan Addison","WR","MIN",6,53.8,4,10.4,23],
    ["Rico Dowdle","RB","CAR",14,54.6,4,10.3,26],
    ["T.J. Hockenson","TE","MIN",6,55.4,3,10.2,28],
    ["Baker Mayfield","QB","TB",9,56.2,2,18.8,30],
    ["Jauan Jennings","WR","SF",9,57.0,4,10.1,28],
    ["Tony Pollard","RB","TEN",10,57.8,4,10.0,28],
    ["Bo Nix","QB","DEN",14,58.5,3,18.4,25],
    ["David Njoku","TE","CLE",9,59.3,3,9.9,29],
    ["D'Andre Swift","RB","CHI",7,60.1,4,9.8,26],
    ["Cooper Kupp","WR","SEA",8,60.9,4,9.7,32],
    ["Jaylen Warren","RB","PIT",9,61.7,4,9.6,26],
    ["Tucker Kraft","TE","GB",10,62.5,3,9.5,24],
    ["Brock Purdy","QB","SF",9,63.2,3,18.0,25],
    ["Stefon Diggs","WR","NE",14,64.0,4,9.4,31],
    ["J.K. Dobbins","RB","DEN",14,64.8,4,9.3,26],
    ["Caleb Williams","QB","CHI",7,65.6,3,17.6,23],
    ["Chris Godwin","WR","TB",9,66.4,4,9.2,29],
    ["Evan Engram","TE","DEN",14,67.2,4,9.1,31],
    ["Rhamondre Stevenson","RB","NE",14,68.0,4,9.0,27],
    ["Ricky Pearsall","WR","SF",9,68.8,4,8.9,24],
    ["Jordan Love","QB","GB",10,69.5,3,17.2,26],
    ["Emeka Egbuka","WR","TB",9,70.3,4,8.8,23],
    ["Dallas Goedert","TE","PHI",9,71.1,4,8.7,30],
    ["Tyrone Tracy","RB","NYG",14,71.9,4,8.6,25],
    ["Drake Maye","QB","NE",14,72.7,3,16.8,23],
    ["Hunter Henry","TE","NE",14,73.5,4,8.5,30],
    ["Najee Harris","RB","LAC",12,74.3,5,8.4,27],
    ["Colston Loveland","TE","CHI",7,75.1,4,8.3,21],
    ["Kyler Murray","QB","ARI",8,75.8,3,16.4,28],
    ["Jakobi Meyers","WR","LV",8,76.6,5,8.2,28],
    ["Tyler Warren","TE","IND",14,77.4,4,8.1,23],
    ["C.J. Stroud","QB","HOU",14,78.2,3,16.1,24],
    ["Keon Coleman","WR","BUF",7,79.0,5,8.0,22],
    ["Bhayshul Tuten","RB","JAC",8,79.8,5,7.9,22],
    ["Jared Goff","QB","DET",8,80.5,4,15.8,31],
    ["Jake Ferguson","TE","DAL",7,81.3,4,7.8,26],
    ["Kaleb Johnson","RB","PIT",9,82.1,5,7.7,22],
    ["Trevor Lawrence","QB","JAC",8,83.0,4,15.5,26],
    ["Michael Penix Jr.","QB","ATL",5,84.2,4,15.1,25],
    ["Isaiah Likely","TE","BAL",7,85.0,5,7.6,25],
    ["Quinshon Judkins","RB","CLE",9,85.8,5,7.5,21]
  ];

  const DEPTH_FIRST = ["Marcus","Andre","Nolan","Isaiah","Calvin","Darius","Miles","Jonah","Ellis","Trevor","Quinton","Brady","Cole","Hayes","Malik","Devin","Roman","Tate","Silas","Owen"];
  const DEPTH_LAST = ["Hendricks","Porter","Brooks","Nguyen","Ellison","Vaughn","Pruitt","Caldwell","Bishop","Ramsey","Hodge","Bennett","Crowder","Lang","Vickers","Morrow","Pritchard","Stanton","Iverson","Crowe"];
  const NFL = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAC","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"];
  const BYE = { ARI:8, ATL:5, BAL:7, BUF:7, CAR:14, CHI:7, CIN:10, CLE:9, DAL:7, DEN:14, DET:8, GB:10, HOU:14, IND:14, JAC:8, KC:10, LAC:12, LAR:8, LV:8, MIA:12, MIN:6, NE:14, NO:12, NYG:14, NYJ:12, PHI:9, PIT:9, SEA:8, SF:9, TB:9, TEN:10, WAS:12 };

  let players = [];
  let byId = {};

  function buildPool() {
    const seen = {};
    players = [];
    RAW.forEach(function (r, i) {
      const key = r[0] + r[2];
      if (seen[key]) return;
      seen[key] = true;
      const p = {
        id: "p" + (players.length + 1),
        name: r[0], pos: r[1], team: r[2], bye: r[3],
        adp: r[4], tier: r[5], ppg: r[6], age: r[7]
      };
      p.val = Math.round(Math.max(8, 168 * Math.exp(-p.adp / 42)));
      players.push(p);
    });
    const posCycle = ["WR", "RB", "WR", "RB", "TE", "WR", "RB", "QB"];
    for (let i = 0; i < 90; i++) {
      const pos = posCycle[i % posCycle.length];
      const team = NFL[i % NFL.length];
      const adp = 84 + i * 1.05;
      const p = {
        id: "d" + i,
        name: DEPTH_FIRST[i % DEPTH_FIRST.length] + " " + DEPTH_LAST[(i * 3) % DEPTH_LAST.length],
        pos: pos, team: team, bye: BYE[team] || 10,
        adp: Math.round(adp * 10) / 10,
        tier: adp < 100 ? 5 : 6,
        ppg: Math.max(3.5, 8.2 - i * 0.035),
        age: 23 + (i % 8)
      };
      p.val = Math.round(Math.max(6, 168 * Math.exp(-p.adp / 42)));
      players.push(p);
    }
    byId = {};
    players.forEach(function (p) { byId[p.id] = p; });
  }
  let byName = {};
  function indexNames() {
    byName = {};
    players.forEach(function (p) { byName[normName(p.name)] = p; });
  }

  const IC = {
    fire: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c2 4-2 6 0 10 3-2 6-1 6 4a6 6 0 1 1-12 0c0-3 2-5 6-14z"/></svg>',
    warn: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 20h20z"/><path d="M12 9v5"/><path d="M12 17h.01"/></svg>',
    cal: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>',
    link: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 1 0 7 7l1-1"/></svg>',
    gem: '<svg class="recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M6 3h12l3 6-9 12L3 9z"/><path d="M3 9h18"/><path d="M9 3 7.5 9 12 21l4.5-12L15 3"/></svg>',
    down: '<svg class="recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 8 9 12 13 9 21 16"/><polyline points="21 11 21 16 16 16"/></svg>',
    bars: '<svg class="recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M5 20V10M12 20V4M19 20v-7"/></svg>',
    trophy: '<svg class="recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M7 6H4.5A1.5 1.5 0 0 0 3 7.5 3.5 3.5 0 0 0 6.5 11M17 6h2.5A1.5 1.5 0 0 1 21 7.5 3.5 3.5 0 0 1 17.5 11M9.5 18h5M8.5 21h7M12 14v4"/></svg>',
    sheet: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M9 12h6M9 16h4"/></svg>',
    room: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M8 4v16"/></svg>'
  };

  const EMBEDDED = (function () {
    try {
      const q = new URLSearchParams(location.search);
      if (q.get("embed") === "1") return true;
      return window.parent !== window;
    } catch (_e) {
      return false;
    }
  })();
  if (EMBEDDED) document.documentElement.classList.add("br-da-embed");

  const state = {
    teams: 12,
    rounds: 15,
    mySlot: 7,
    live: EMBEDDED,
    teamNames: {},
    current: 1,
    picks: [],
    drafted: {},
    auto: false,
    tab: "board",
    pos: "ALL",
    sort: "rec",
    platform: "sleeper",
    clock: CLOCK_START,
    expanded: null,
    toast: "",
    syncOk: true,
    valCap: 180,
    sitePool: false,
    adpSource: "consensus",
    adpOptions: [
      { value: "consensus", label: "Consensus" },
      { value: "sleeper", label: "Sleeper" },
      { value: "espn", label: "ESPN" },
      { value: "yahoo", label: "Yahoo" },
      { value: "mfl", label: "MFL" },
      { value: "brfantasy", label: "BR Fantasy" }
    ]
  };
  let autoTimer = null, clockTimer = null;
  let lastLiveDetail = null;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }
  function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
  function ownerOf(pn) {
    const n = state.teams;
    const r = Math.ceil(pn / n);
    const i = (pn - 1) % n;
    return (r % 2 === 1) ? (i + 1) : (n - i);
  }
  function isMine(pn) { return ownerOf(pn) === state.mySlot; }
  function nextMine(from) {
    for (let pn = from; pn <= state.teams * state.rounds; pn++) if (isMine(pn)) return pn;
    return null;
  }
  function pickLabel(pn) {
    const n = state.teams;
    const rd = Math.ceil(pn / n);
    const pk = pn - (rd - 1) * n;
    return rd + "." + String(pk).padStart(2, "0");
  }
  function draftDone() { return state.current > state.teams * state.rounds; }
  function available() { return players.filter(function (p) { return !state.drafted[p.id]; }); }
  function teamPicks(slot) {
    return state.picks.filter(function (x) { return x.slot === slot; }).map(function (x) { return Object.assign({}, x.p, { pn: x.pn, grade: x.grade }); });
  }
  function myPicks() { return teamPicks(state.mySlot); }
  function teamName(slot) {
    if (slot === state.mySlot) return "You";
    if (state.teamNames[slot]) return state.teamNames[slot];
    return TEAM_NAMES[slot] || ("Team " + slot);
  }
  function normName(s) {
    return String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "").replace(/(jr|sr|ii|iii|iv)$/, "");
  }
  function posCounts(list) {
    const c = { QB: 0, RB: 0, WR: 0, TE: 0 };
    list.forEach(function (p) { if (c[p.pos] != null) c[p.pos]++; });
    return c;
  }
  function needOf(counts, pos) {
    const t = { QB: 1, RB: 3, WR: 3, TE: 1 };
    return Math.max(0, (t[pos] || 0) - (counts[pos] || 0));
  }
  function slotColor(s) { return POS[s] || "#64748b"; }
  function psColor(ps) { return ps >= 90 ? "#22c55e" : ps >= 75 ? "#38bdf8" : ps >= 60 ? "#f59e0b" : "#ef4444"; }
  function gradeCol(s) { return s >= 75 ? "#22c55e" : s >= 60 ? "#38bdf8" : s >= 45 ? "#f59e0b" : "#ef4444"; }
  function gradeLetter(s) {
    if (s >= 90) return "A+"; if (s >= 85) return "A"; if (s >= 80) return "A-";
    if (s >= 75) return "B+"; if (s >= 70) return "B"; if (s >= 65) return "B-";
    if (s >= 60) return "C+"; if (s >= 55) return "C"; if (s >= 50) return "C-";
    if (s >= 40) return "D"; return "F";
  }
  function pickLetter(diff, need) {
    let s = 55;
    if (diff >= 12) s = 92; else if (diff >= 6) s = 86; else if (diff >= 2) s = 78;
    else if (diff >= -2) s = 72; else if (diff >= -6) s = 62; else if (diff >= -12) s = 48; else s = 32;
    if (need) s += 6;
    return gradeLetter(clamp(s, 20, 98));
  }

  function pickScore(p, counts, pickNo) {
    const adp = p.adp;
    const rel = (pickNo - adp) / Math.max(adp, 1.5);
    let adpVal = rel >= 0.5 ? 1 : rel >= -0.3 ? 0.5 + rel : Math.max(0, 0.2 + rel * 0.25);
    const need = needOf(counts, p.pos);
    const needN = clamp(need / 2, 0, 1);
    const ppgN = clamp((p.ppg - 6) / 16, 0, 1);
    const tierN = clamp((7 - p.tier) / 6, 0, 1);
    const valN = clamp(p.val / Math.max(state.valCap || 180, 1), 0, 1);
    let s = 100 * (0.28 * adpVal + 0.24 * valN + 0.18 * ppgN + 0.16 * tierN + 0.14 * needN);
    if (isTierCliff(p) && need > 0) s += 4;
    return Math.round(clamp(s, 8, 99));
  }
  function decisionScore(p, counts, pickNo) {
    const ps = pickScore(p, counts, pickNo);
    const need = needOf(counts, p.pos);
    let ds = ps + need * 8 + (p.tier <= 2 ? 4 : 0);
    if (!need && p.pos === "QB" && (counts.QB || 0) >= 1) ds -= 18;
    if (!need && p.pos === "TE" && (counts.TE || 0) >= 1) ds -= 8;
    return ds;
  }
  function rankedPool(counts, pickNo) {
    const pool = available().map(function (p) {
      return Object.assign({}, p, { _ps: pickScore(p, counts, pickNo), _ds: decisionScore(p, counts, pickNo) });
    });
    pool.sort(function (a, b) { return (b._ds - a._ds) || (b._ps - a._ps) || (a.adp - b.adp); });
    pool.forEach(function (p, i) { p._rank = i + 1; });
    return pool;
  }
  function isTierCliff(p) {
    const left = available().filter(function (q) { return q.pos === p.pos && q.tier === p.tier; }).length;
    return p.tier <= 2 && left <= 2 && state.current > state.teams;
  }
  function reasonsFor(p, counts, pickNo) {
    const out = [];
    const need = needOf(counts, p.pos);
    const diff = Math.round(pickNo - p.adp);
    if (need > 0) out.push("Fills a starting " + p.pos + " need (" + need + " still open)");
    else if (p.pos !== "QB" && (counts.RB + counts.WR + counts.TE) < 5) out.push("FLEX-eligible depth for weekly lineup");
    if (diff >= 4) out.push("Value vs ADP: " + diff + " picks past market");
    else if (diff <= -4) out.push("Slight reach vs ADP (" + Math.abs(diff) + " early) — still a positional fit");
    else out.push("In range of ADP " + fmtAdp(p));
    if (isTierCliff(p)) out.push("Tier cliff: last " + p.pos + "s in T" + p.tier);
    else if (p.tier <= 2) out.push("Elite tier (T" + p.tier + ") talent still on the board");
    return out.slice(0, 3);
  }

  function cpuChoose(slot, avoidId) {
    const counts = posCounts(teamPicks(slot));
    const pool = rankedPool(counts, state.current).filter(function (p) { return p.id !== avoidId; });
    if (!pool.length) return null;
    const n = Math.min(4, pool.length);
    const w = [];
    let tot = 0;
    for (let i = 0; i < n; i++) { const wt = (n - i) * (n - i); w.push(wt); tot += wt; }
    let r = Math.random() * tot;
    for (let i = 0; i < n; i++) { r -= w[i]; if (r <= 0) return pool[i]; }
    return pool[0];
  }

  function commitPick(player, slot) {
    if (!player || draftDone() || state.drafted[player.id]) return;
    const pn = state.current;
    const counts = posCounts(teamPicks(slot));
    const need = needOf(counts, player.pos) > 0;
    const grade = pickLetter(pn - player.adp, need);
    const ps = pickScore(player, counts, pn);
    state.picks.push({ pn: pn, slot: slot, p: player, grade: grade, ps: ps });
    state.drafted[player.id] = true;
    state.current += 1;
    state.clock = CLOCK_START;
  }

  function simulateOne() {
    if (draftDone()) return;
    const slot = ownerOf(state.current);
    const counts = posCounts(teamPicks(slot));
    const pool = rankedPool(counts, state.current);
    const pick = slot === state.mySlot ? pool[0] : cpuChoose(slot);
    if (pick) commitPick(pick, slot);
  }

  function draftToMe(id) {
    if (draftDone()) return;
    const player = byId[id];
    if (!player || state.drafted[id]) return;
    while (!draftDone() && !isMine(state.current)) {
      const slot = ownerOf(state.current);
      const pick = cpuChoose(slot, id);
      if (pick) commitPick(pick, slot); else break;
    }
    if (!draftDone() && isMine(state.current) && !state.drafted[id]) {
      commitPick(player, state.mySlot);
    }
    while (!draftDone() && !isMine(state.current)) {
      const slot = ownerOf(state.current);
      const pick = cpuChoose(slot);
      if (pick) commitPick(pick, slot); else break;
    }
  }

  function resetDraft() {
    state.current = 1;
    state.picks = [];
    state.drafted = {};
    state.clock = CLOCK_START;
    state.expanded = null;
    state.toast = "";
    stopAuto();
  }

  function optimalLineup(list) {
    const leftover = list.slice().sort(function (a, b) { return b.ppg - a.ppg; });
    const starters = [];
    function take(slot, ok) {
      const i = leftover.findIndex(ok);
      if (i < 0) { starters.push({ slot: slot, p: null }); return; }
      starters.push({ slot: slot, p: leftover.splice(i, 1)[0] });
    }
    take("QB", function (p) { return p.pos === "QB"; });
    take("RB", function (p) { return p.pos === "RB"; });
    take("RB", function (p) { return p.pos === "RB"; });
    take("WR", function (p) { return p.pos === "WR"; });
    take("WR", function (p) { return p.pos === "WR"; });
    take("TE", function (p) { return p.pos === "TE"; });
    take("FLEX", function (p) { return p.pos === "RB" || p.pos === "WR" || p.pos === "TE"; });
    return { starters: starters, bench: leftover };
  }

  function competitiveWindow(list) {
    const ol = optimalLineup(list);
    let wSum = 0, aSum = 0;
    ol.starters.forEach(function (x) {
      if (!x.p) return;
      const w = Math.max(1, x.p.val || 1);
      aSum += x.p.age * w; wSum += w;
    });
    if (wSum <= 0) return { label: "Balanced", avgAge: 0 };
    const avgAge = aSum / wSum;
    const label = avgAge <= 24.5 ? "Future" : avgAge >= 26.5 ? "Win-Now" : "Balanced";
    return { label: label, avgAge: avgAge };
  }

  function rawTeamScore(slot) {
    const list = teamPicks(slot);
    const n = list.length;
    if (!n) return { score: 70, value: 10, starters: 25, construction: 15, provisional: true };
    let adpPts = 0;
    list.forEach(function (p) {
      const d = p.pn - p.adp;
      adpPts += clamp(50 + d * 3.2, 10, 98);
    });
    const value = adpPts / n;
    const ol = optimalLineup(list);
    const filled = ol.starters.filter(function (x) { return x.p; }).length;
    const coverage = filled / SLOTS.length;
    const counts = posCounts(list);
    const balance = 100 * coverage * 0.7 + 30 * (1 - Math.abs((counts.RB || 0) - (counts.WR || 0)) / 8);
    const stars = list.slice().sort(function (a, b) { return b.ppg - a.ppg; }).slice(0, 3)
      .reduce(function (s, p) { return s + p.ppg; }, 0);
    const starN = clamp(stars / 52 * 100, 15, 98);
    const construction = clamp(balance, 12, 98);
    const blend = 0.40 * value + 0.30 * starN + 0.30 * construction;
    return {
      score: blend,
      value: value,
      starters: starN,
      construction: construction,
      coverage: coverage,
      provisional: n < 8,
      count: n
    };
  }

  function gradeAllTeams() {
    const raw = [];
    for (let s = 1; s <= state.teams; s++) raw.push(rawTeamScore(s));
    const scores = raw.map(function (g) { return g.score; });
    const mean = scores.reduce(function (a, b) { return a + b; }, 0) / state.teams;
    const varr = scores.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / state.teams;
    const sd = Math.max(3.5, Math.sqrt(varr));
    return raw.map(function (g, i) {
      const z = (g.score - mean) / sd;
      let curved = 70 + z * 9.5;
      const damp = Math.min(1, (g.count || 0) / 8);
      const score = clamp(damp * curved + (1 - damp) * 70, 22, 97);
      return {
        slot: i + 1,
        name: teamName(i + 1),
        isMe: i + 1 === state.mySlot,
        picks: teamPicks(i + 1),
        grade: {
          score: score,
          value: g.value,
          starters: g.starters,
          construction: g.construction,
          provisional: g.provisional,
          window: competitiveWindow(teamPicks(i + 1))
        }
      };
    }).sort(function (a, b) { return b.grade.score - a.grade.score; });
  }

  function playoffOdds(all) {
    if (!draftDone()) return {};
    const str = all.map(function (t) {
      const ol = optimalLineup(t.picks);
      let s = 0;
      ol.starters.forEach(function (x) { if (x.p) s += x.p.ppg; });
      return { slot: t.slot, S: s };
    });
    const SIMS = 360, SPOTS = 6, N = 14, sigma = 7.5;
    const hits = {};
    str.forEach(function (t) { hits[t.slot] = 0; });
    function gauss() {
      let u = 0, v = 0;
      while (!u) u = Math.random();
      while (!v) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    for (let i = 0; i < SIMS; i++) {
      const season = str.map(function (t) {
        return { slot: t.slot, pts: t.S * N + gauss() * sigma * Math.sqrt(N) };
      }).sort(function (a, b) { return b.pts - a.pts; });
      for (let k = 0; k < SPOTS; k++) hits[season[k].slot]++;
    }
    const out = {};
    str.forEach(function (t) { out[t.slot] = Math.round(1000 * hits[t.slot] / SIMS) / 10; });
    return out;
  }

  function bannersHtml(counts, rec) {
    let html = "";
    const last = state.picks.slice(-5);
    const hot = { QB: 0, RB: 0, WR: 0, TE: 0 };
    last.forEach(function (x) { if (hot[x.p.pos] != null) hot[x.p.pos]++; });
    let run = "";
    ["RB", "WR", "QB", "TE"].forEach(function (pos) { if (!run && hot[pos] >= 3) run = pos; });
    if (run) {
      html += '<div class="banner banner-run">' + IC.fire + '<span><b>' + run + " run</b>: " + hot[run] + " of the last 5 picks. Weigh your " + run + " need before the tier dries up.</span></div>";
    }
    if (state.current > state.teams) {
      ["QB", "RB", "WR", "TE"].forEach(function (pos) {
        const n = available().filter(function (p) { return p.pos === pos && p.tier <= 2; }).length;
        if (n === 1) html += '<div class="banner banner-cliff">' + IC.warn + "<span><b>Last T1-2 " + pos + "</b> on the board.</span></div>";
        else if (n === 2) html += '<div class="banner banner-cliff">' + IC.warn + "<span><b>Only 2 T1-2 " + pos + "s</b> left.</span></div>";
      });
    }
    const mine = myPicks();
    const byeMap = {};
    mine.forEach(function (p) { byeMap[p.bye] = (byeMap[p.bye] || 0) + 1; });
    let stackedBye = null;
    Object.keys(byeMap).forEach(function (b) { if (byeMap[b] >= 2) stackedBye = +b; });
    if (stackedBye && rec && rec.bye === stackedBye) {
      html += '<div class="banner banner-bye">' + IC.cal + "<span><b>Bye stack</b>: you already have " + byeMap[stackedBye] + " on Bye " + stackedBye + ". " + esc(rec.name) + " shares it.</span></div>";
    }
    const myQb = mine.filter(function (p) { return p.pos === "QB"; })[0];
    if (myQb) {
      const stack = available().filter(function (p) { return (p.pos === "WR" || p.pos === "TE") && p.team === myQb.team && p.adp <= 80; })[0];
      if (stack) {
        html += '<div class="banner banner-stack">' + IC.link + "<span><b>QB-stack opportunity</b>: " + esc(stack.name) + " (" + stack.pos + ") still available with " + esc(myQb.name) + ".</span></div>";
      }
    }
    return html;
  }

  function fmtAdp(p) {
    const a = Number(p && p.adp);
    if (!isFinite(a) || a >= 900) return "—";
    return a.toFixed(1);
  }
  function fmtPpg(p) {
    const n = Number(p && p.ppg);
    if (!isFinite(n) || n <= 0) return "—";
    return n.toFixed(1);
  }
  function hsUrl(p) {
    if (!p) return "";
    if (p.headshot) return String(p.headshot);
    const id = String(p.id || "");
    if (/^\d+$/.test(id)) return "https://sleepercdn.com/content/nfl/players/" + id + ".jpg";
    return "";
  }
  function hsMark(p, cls) {
    const pc = POS[p.pos] || POS.WR;
    const url = hsUrl(p);
    if (!url) return '<span class="' + cls + '" style="--pc:' + pc + '" aria-hidden="true"></span>';
    const fallback = /^\d+$/.test(String(p.id || ""))
      ? "https://sleepercdn.com/content/nfl/players/" + p.id + ".jpg"
      : "";
    const extra = fallback && fallback !== url ? ' data-fallback="' + esc(fallback) + '"' : "";
    return '<span class="' + cls + ' has-photo" style="--pc:' + pc + '" aria-hidden="true"><img alt="" src="' + esc(url) + '"' + extra + "></span>";
  }

  function playerRow(p, opts) {
    opts = opts || {};
    const cliff = isTierCliff(p);
    const pc = POS[p.pos];
    const mine = myPicks();
    const byeLvl = mine.filter(function (x) { return x.bye === p.bye; }).length >= 2;
    let chip;
    if (state.sort === "rec") {
      chip = '<div class="pschip recchip" title="Recommendation rank">#' + (opts.rank || p._rank) + "<small>REC</small></div>";
    } else {
      const col = psColor(p._ps);
      chip = '<div class="pschip" style="color:' + col + ";background:" + col + '1a" title="Pick Score">'+ p._ps + "<small>PS</small></div>";
    }
    return '<div class="ba-row" data-id="' + p.id + '">'
      + hsMark(p, "hs")
      + '<div class="ba-body"><div class="ba-name">' + esc(p.name) + "</div>"
      + '<div class="ba-meta"><span class="posb" style="background:' + pc + '">' + p.pos + "</span>"
      + esc(p.team)
      + '<span class="tier' + (cliff ? " cliff" : "") + '">T' + p.tier + "</span>"
      + '<span class="tabular">' + fmtPpg(p) + " proj</span>"
      + (byeLvl ? '<span class="bye-flag">Bye ' + p.bye + "</span>" : "")
      + "</div></div>"
      + '<div class="ba-right"><div class="ba-val">' + p.val + '</div><div class="ba-sub">ADP ' + fmtAdp(p) + "</div></div>"
      + chip
      + "</div>";
  }

  function slotRow(slot, p) {
    if (p) {
      const g = p.grade || pickLetter((p.pn || 0) - p.adp, true);
      return '<div class="rslot">'
        + '<span class="rslot-pos" style="background:' + slotColor(slot) + '">' + slot + "</span>"
        + hsMark(p, "hs-sm")
        + '<div class="rslot-body"><div class="rslot-name">' + esc(p.name) + "</div>"
        + '<div class="rslot-meta">' + p.pos + " · " + p.team + (p.pn ? ' · <span class="tabular">' + pickLabel(p.pn) + "</span>" : "") + "</div></div>"
        + '<span class="rslot-g" style="color:' + gradeCol(letterToScore(g)) + '">' + g + "</span>"
        + '<span class="rslot-val">' + p.val + "</span></div>";
    }
    return '<div class="rslot open"><span class="rslot-pos" style="background:' + slotColor(slot) + '">' + slot + '</span><span class="rslot-empty">open</span></div>';
  }
  function rankMedal(rank) {
    if (rank > 3) return '<span class="lrank">' + rank + "</span>";
    const metals = {
      1: { face: "#f6d375", rim: "#c99a2e", num: "#8a6410" },
      2: { face: "#dde3ea", rim: "#aab4bf", num: "#5c6670" },
      3: { face: "#e0a56a", rim: "#b87a44", num: "#7a4620" }
    };
    const m = metals[rank];
    const cls = ["gold", "silver", "bronze"][rank - 1];
    return '<span class="lrank has-medal ' + cls + '"><svg viewBox="0 0 24 24" width="20" height="20" aria-label="Rank ' + rank + '">'
      + '<circle cx="12" cy="12" r="10" fill="' + m.rim + '"/>'
      + '<circle cx="12" cy="12" r="8" fill="' + m.face + '"/>'
      + '<text x="12" y="16" text-anchor="middle" font-size="10" font-weight="800" fill="' + m.num + '" font-family="Archivo,sans-serif">' + rank + "</text></svg></span>";
  }
  function letterToScore(letter) {
    return { "A+": 92, A: 87, "A-": 82, "B+": 77, B: 72, "B-": 67, "C+": 62, C: 57, "C-": 52, D: 43, F: 20 }[letter] || 55;
  }

  function renderBoard() {
    if (EMBEDDED && !state.sitePool) {
      return '<div class="empty-log">Loading BR Fantasy ranks, ADP, and values…</div>';
    }
    const mine = myPicks();
    const counts = posCounts(mine);
    const recPn = nextMine(state.current) || state.current;
    const pool = rankedPool(counts, recPn);
    let html = "";
    if (draftDone()) {
      const all = gradeAllTeams();
      const me = all.filter(function (t) { return t.isMe; })[0];
      const rank = all.findIndex(function (t) { return t.isMe; }) + 1;
      const odds = playoffOdds(all)[state.mySlot];
      html += '<div class="rec-card final-grade"><div class="rec-label">Draft complete</div>'
        + '<div class="letter" style="color:' + gradeCol(me.grade.score) + '">' + gradeLetter(me.grade.score) + "</div>"
        + '<div class="rank">League rank #' + rank + " of " + state.teams + "</div>"
        + '<div class="sub">Projected playoff odds <b class="tabular" style="color:' + (odds >= 50 ? "var(--win)" : "var(--warn)") + '">' + odds.toFixed(1) + "%</b></div></div>";
    } else if (pool[0]) {
      const rec = pool[0];
      const col = psColor(rec._ps);
      html += '<div class="rec-card"><div class="rec-label">Recommended pick</div><div class="rec-top">'
        + '<div class="gauge" style="--p:' + rec._ps + ";--g:" + col + '"><div><b>' + rec._ps + "</b><small>PS</small></div></div>"
        + '<div class="rec-player">' + hsMark(rec, "hs") + '<div><div class="rec-name">' + esc(rec.name) + "</div>"
        + '<div class="rec-meta"><span class="posb" style="background:' + POS[rec.pos] + '">' + rec.pos + "</span>"
        + rec.team + " · T" + rec.tier + " · " + fmtPpg(rec) + " proj</div></div></div>"
        + '<ul class="reasons">' + reasonsFor(rec, counts, recPn).map(function (r) { return "<li>" + esc(r) + "</li>"; }).join("") + "</ul>"
        + '<button type="button" class="btn btn-primary draft-cta" data-draft="' + rec.id + '">'
        + (state.live ? ("Recommend " + esc(rec.name) + " at " + pickLabel(recPn)) : ("Draft " + esc(rec.name) + " at " + pickLabel(recPn)))
        + "</button></div>";
      html += bannersHtml(counts, rec);
    }
    html += '<div class="filters">';
    ["ALL", "QB", "RB", "WR", "TE"].forEach(function (pos) {
      html += '<button type="button" class="chip" data-pos="' + pos + '" aria-pressed="' + (state.pos === pos) + '">' + (pos === "ALL" ? "All" : pos) + "</button>";
    });
    html += '<button type="button" class="chip sort-btn" id="sortBtn">Sort: ' + SORT_LBL[state.sort] + "</button></div>";
    let rows = pool;
    if (state.pos !== "ALL") rows = rows.filter(function (p) { return p.pos === state.pos; });
    if (state.sort === "adp") rows = rows.slice().sort(function (a, b) { return a.adp - b.adp; });
    else if (state.sort === "ps") rows = rows.slice().sort(function (a, b) { return b._ps - a._ps; });
    else if (state.sort === "proj") rows = rows.slice().sort(function (a, b) { return b.ppg - a.ppg; });
    rows.slice(0, 40).forEach(function (p, i) {
      html += playerRow(p, { rank: state.sort === "rec" ? p._rank : i + 1 });
    });
    if (!rows.length) html += '<div class="empty-log" style="color:var(--text-muted)">No players match this filter.</div>';
    return html;
  }

  function gbar(label, val) {
    const pct = Math.round(clamp(val, 0, 100));
    const col = pct >= 80 ? "#22c55e" : pct >= 60 ? "#38bdf8" : pct >= 40 ? "#f59e0b" : "#ef4444";
    return '<div class="gbar-row"><span class="gbar-lbl">' + label + "</span>"
      + '<div class="gbar"><div class="gbar-fill" style="width:' + pct + "%;background:" + col + '"></div></div>'
      + '<span class="gbar-pct" style="color:' + col + '">' + pct + "</span></div>";
  }

  function renderRoster() {
    const mine = myPicks();
    const all = gradeAllTeams();
    const me = all.filter(function (t) { return t.isMe; })[0];
    const g = me.grade;
    let html = '<div class="grade-card"><div><div class="grade-letter" style="color:' + gradeCol(g.score) + '">' + gradeLetter(g.score) + "</div>"
      + (g.provisional ? '<div class="grade-early">Early</div>' : "") + "</div>"
      + '<div class="grade-meta">' + gbar("Value", g.value) + gbar("Starters", g.starters) + gbar("Construction", g.construction) + "</div></div>";

    if (mine.length >= 1) {
      const myAvg = mine.reduce(function (s, p) { return s + p.ppg; }, 0) / mine.length;
      const draftedP = state.picks.map(function (x) { return x.p.ppg; });
      const lgAvg = draftedP.length ? draftedP.reduce(function (a, b) { return a + b; }, 0) / draftedP.length : myAvg;
      const pct = lgAvg > 0 ? Math.round(myAvg / lgAvg * 100) : 100;
      const col = pct >= 108 ? "#22c55e" : pct >= 92 ? "#f59e0b" : "#ef4444";
      html += '<div class="proj-card"><div class="proj-title">Roster Projection</div><div class="proj-stats">'
        + '<div class="proj-stat"><div class="proj-val">' + myAvg.toFixed(1) + '</div><div class="proj-lbl">My Avg PPG</div></div>'
        + '<div class="proj-stat"><div class="proj-val">' + lgAvg.toFixed(1) + '</div><div class="proj-lbl">Avg Player</div></div>'
        + '<div class="proj-stat"><div class="proj-val" style="color:' + col + '">' + pct + '%</div><div class="proj-lbl">vs League</div></div>'
        + '</div><div class="proj-bar"><div class="gbar-fill" style="width:' + Math.min(100, pct) + "%;background:" + col + '"></div></div></div>';
    }
    const w = g.window;
    const wcls = w.label === "Future" ? "win-future" : w.label === "Win-Now" ? "win-winnow" : "win-balanced";
    html += '<div class="win-row"><span class="win-chip ' + wcls + '">' + w.label + "</span>";
    if (w.avgAge) html += '<span style="font-size:11px;color:var(--text-muted)">Avg age ' + w.avgAge.toFixed(1) + "</span>";
    if (draftDone()) {
      const odds = playoffOdds(all)[state.mySlot];
      html += '<span class="odds-chip" style="color:' + (odds >= 50 ? "var(--win)" : "var(--warn)") + '">' + odds.toFixed(1) + "% playoff</span>";
    }
    html += "</div>";

    const ol = optimalLineup(mine);
    html += '<div class="roster">';
    ol.starters.forEach(function (s) { html += slotRow(s.slot, s.p); });
    html += '<div class="roster-div">Bench</div>';
    if (ol.bench.length) ol.bench.forEach(function (p) { html += slotRow("BN", p); });
    else html += slotRow("BN", null);
    html += "</div>";
    if (state.toast) html += '<div class="toast">' + esc(state.toast) + "</div>";
    html += '<div class="deeplinks">'
      + '<button type="button" data-link="room">' + IC.room + " Draft Room</button>"
      + '<button type="button" data-link="sheet">' + IC.sheet + " Cheat Sheet</button></div>";
    return html;
  }

  function recapHtml(all) {
    const rows = state.picks.map(function (x) {
      return { name: x.p.name, pos: x.p.pos, team: teamName(x.slot), pn: x.pn, gap: x.pn - x.p.adp, teamSlot: x.slot };
    });
    if (rows.length < 4) return "";
    const steals = rows.slice().sort(function (a, b) { return b.gap - a.gap; }).slice(0, 3);
    const reaches = rows.slice().sort(function (a, b) { return a.gap - b.gap; }).slice(0, 3);
    function line(x, good) {
      const txt = (x.gap >= 0 ? "+" : "") + Math.round(x.gap);
      return '<div class="recap-row"><span class="posb" style="background:' + POS[x.pos] + '">' + x.pos + "</span>"
        + '<span class="recap-main"><span class="recap-name">' + esc(x.name) + "</span>"
        + '<span class="recap-sub">' + esc(x.team) + " · " + pickLabel(x.pn) + "</span></span>"
        + '<span class="recap-gap" style="color:' + (good ? "var(--win)" : "var(--loss)") + '">' + txt + "</span></div>";
    }
    let bestT = all[0], bestAvg = -1e9;
    all.forEach(function (t) {
      if (!t.picks.length) return;
      const avg = t.picks.reduce(function (s, p) { return s + (p.pn - p.adp); }, 0) / t.picks.length;
      if (avg > bestAvg) { bestAvg = avg; bestT = t; }
    });
    const posCount = {};
    state.picks.forEach(function (x) { posCount[x.p.pos] = (posCount[x.p.pos] || 0) + 1; });
    const topPos = Object.keys(posCount).sort(function (a, b) { return posCount[b] - posCount[a]; })[0] || "—";
    return '<div class="recap"><div><p class="recap-h">' + IC.gem + "Biggest steals</p>" + steals.map(function (x) { return line(x, true); }).join("")
      + '</div><div><p class="recap-h">' + IC.down + "Biggest reaches</p>" + reaches.map(function (x) { return line(x, false); }).join("")
      + "</div></div>"
      + '<p class="recap-h" style="padding:4px 12px 0">' + IC.bars + "By the numbers</p>"
      + '<div class="nums"><div class="tile"><div class="tile-l">Steal of the draft</div><div class="tile-b">' + esc(steals[0].name) + '</div><div class="tile-s">' + esc(steals[0].team) + " · " + pickLabel(steals[0].pn) + "</div></div>"
      + '<div class="tile"><div class="tile-l">Biggest reach</div><div class="tile-b">' + esc(reaches[0].name) + '</div><div class="tile-s">' + esc(reaches[0].team) + " · " + pickLabel(reaches[0].pn) + "</div></div>"
      + '<div class="tile"><div class="tile-l">Best value drafter</div><div class="tile-b">' + esc(bestT.name) + '</div><div class="tile-s">Highest average ADP gap</div></div>'
      + '<div class="tile"><div class="tile-l">Most drafted</div><div class="tile-b">' + topPos + " (" + posCount[topPos] + ')</div><div class="tile-s">' + state.picks.length + " picks total</div></div></div>";
  }

  function renderGrades() {
    const all = gradeAllTeams();
    const odds = draftDone() ? playoffOdds(all) : {};
    let html = recapHtml(all);
    html += '<p class="recap-h" style="padding:8px 12px 6px">' + IC.trophy + "Draft grades</p>";
    all.forEach(function (t, i) {
      const w = t.grade.window;
      const wcls = w.label === "Future" ? "win-future" : w.label === "Win-Now" ? "win-winnow" : "win-balanced";
      const open = state.expanded === t.slot;
      html += '<div class="lrow' + (t.isMe ? " is-me" : "") + (open ? " is-open" : "") + '" data-legslot="' + t.slot + '">'
        + rankMedal(i + 1)
        + '<span class="lname">' + esc(t.name) + "</span>"
        + '<span class="win-chip ' + wcls + '">' + w.label + "</span>"
        + (odds[t.slot] != null ? '<span class="lpo" style="color:' + (odds[t.slot] >= 50 ? "var(--win)" : "var(--text-muted)") + '">' + odds[t.slot].toFixed(0) + "%</span>" : "")
        + '<span class="lgrade" style="color:' + gradeCol(t.grade.score) + '">' + gradeLetter(t.grade.score) + "</span>"
        + '<span class="lchev" aria-hidden="true"><svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M2 4l4 4 4-4"/></svg></span></div>';
      html += '<div class="ldtl' + (open ? " is-open" : "") + '" data-dtl="' + t.slot + '">';
      if (open) {
        const ol = optimalLineup(t.picks);
        ol.starters.forEach(function (s) {
          if (!s.p) {
            html += '<div class="ldtl-row"><span class="ldtl-slot" style="background:' + slotColor(s.slot) + '">' + s.slot + '</span><span class="ldtl-name" style="color:var(--text-muted)">open</span></div>';
            return;
          }
          html += '<div class="ldtl-row"><span class="ldtl-slot" style="background:' + slotColor(s.slot) + '">' + s.slot + "</span>"
            + '<span class="ldtl-name">' + esc(s.p.name) + "</span>"
            + (s.p.pn ? '<span class="ldtl-pick">' + pickLabel(s.p.pn) + "</span>" : "") + "</div>";
        });
        ol.bench.forEach(function (p) {
          html += '<div class="ldtl-row"><span class="ldtl-slot" style="background:' + POS.BN + '">BN</span>'
            + '<span class="ldtl-name">' + esc(p.name) + "</span>"
            + (p.pn ? '<span class="ldtl-pick">' + pickLabel(p.pn) + "</span>" : "") + "</div>";
        });
        if (!t.picks.length) html += '<div class="ldtl-row"><span class="ldtl-name" style="color:var(--text-muted)">No picks yet</span></div>';
      }
      html += "</div>";
    });
    return html;
  }

  function renderHost() {
    const pf = PLATFORMS[state.platform];
    document.getElementById("urlText").textContent = pf.url;
    document.getElementById("hostLogo").textContent = pf.logo;
    document.getElementById("hostLeague").textContent = pf.league;
    document.getElementById("syncChip").innerHTML = "<i></i> " + pf.sync;
    const rd = Math.min(state.rounds, Math.ceil(Math.min(state.current, state.teams * state.rounds) / state.teams));
    document.getElementById("hostSub").textContent = "12-team PPR · snake · round " + rd;
    const otc = document.getElementById("otc");
    const done = draftDone();
    const slot = done ? null : ownerOf(state.current);
    otc.classList.toggle("is-you", !done && slot === state.mySlot);
    const m = Math.floor(state.clock / 60), s = state.clock % 60;
    document.getElementById("otcClock").textContent = done ? "0:00" : (m + ":" + String(s).padStart(2, "0"));
    document.getElementById("otcWho").textContent = done ? "Draft complete" : ("On the clock: " + teamName(slot));
    const nxt = done ? null : ownerOf(Math.min(state.current + 1, state.teams * state.rounds));
    document.getElementById("otcNext").textContent = done ? ("All " + state.rounds + " rounds in the books") : ("Next up: " + teamName(nxt));
    document.getElementById("otcPick").textContent = done ? "Final" : pickLabel(state.current);

    const log = document.getElementById("pickLog");
    const recent = state.picks.slice().reverse().slice(0, 24);
    if (!recent.length) {
      log.innerHTML = '<div class="empty-log">Waiting on pick 1.01. Simulate or hit Auto — the overlay never submits to the host.</div>';
    } else {
      log.innerHTML = recent.map(function (x) {
        const mine = x.slot === state.mySlot;
        const gcol = gradeCol(letterToScore(x.grade));
        return '<div class="pick-row' + (mine ? " is-mine" : "") + '">'
          + '<span class="pick-pn">' + pickLabel(x.pn) + "</span>"
          + '<span class="posb" style="background:' + POS[x.p.pos] + '">' + x.p.pos + "</span>"
          + '<span class="pick-name">' + esc(x.p.name) + "</span>"
          + '<span class="pick-team">' + x.p.team + " · " + teamName(x.slot) + "</span>"
          + (mine ? '<span class="grade-badge" style="color:' + gcol + '">' + x.grade + "</span>" : "")
          + "</div>";
      }).join("");
    }
  }

  function renderOverlay() {
    const all = gradeAllTeams();
    const me = all.filter(function (t) { return t.isMe; })[0];
    document.getElementById("rosterChip").textContent = String(myPicks().length);
    const letter = gradeLetter(me.grade.score);
    const chip = document.getElementById("gradesChip");
    chip.textContent = myPicks().length ? letter : "—";
    chip.style.color = myPicks().length ? gradeCol(me.grade.score) : "";
    document.querySelectorAll(".tab-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-tab") === state.tab);
    });
    const body = document.getElementById("ovBody");
    if (state.tab === "roster") body.innerHTML = renderRoster();
    else if (state.tab === "grades") body.innerHTML = renderGrades();
    else body.innerHTML = renderBoard();
    const simBtn = document.getElementById("simBtn");
    if (simBtn) simBtn.disabled = draftDone() || state.live;
  }

  function render() {
    document.getElementById("stage").setAttribute("data-platform", state.platform);
    renderHost();
    renderOverlay();
  }

  function startAuto() {
    state.auto = true;
    document.getElementById("autoBtn").setAttribute("aria-pressed", "true");
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(function () {
      if (draftDone()) { stopAuto(); render(); return; }
      simulateOne();
      render();
    }, 700);
  }
  function stopAuto() {
    state.auto = false;
    document.getElementById("autoBtn").setAttribute("aria-pressed", "false");
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  }

  function applyTheme(mode) {
    document.documentElement.setAttribute("data-theme", mode);
    localStorage.setItem("br-da-theme", mode);
    document.querySelectorAll("[data-theme-opt]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-theme-opt") === mode ? "true" : "false");
    });
  }

  document.querySelectorAll("[data-theme-opt]").forEach(function (b) {
    b.addEventListener("click", function () { applyTheme(b.getAttribute("data-theme-opt")); });
  });
  document.querySelectorAll(".platform-switch [data-platform]").forEach(function (b) {
    b.addEventListener("click", function () {
      state.platform = b.getAttribute("data-platform");
      document.querySelectorAll(".platform-switch [data-platform]").forEach(function (x) {
        x.setAttribute("aria-selected", x === b ? "true" : "false");
      });
      render();
    });
  });
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      state.tab = b.getAttribute("data-tab");
      state.toast = "";
      render();
    });
  });
  document.getElementById("simBtn").addEventListener("click", function () {
    simulateOne();
    render();
  });
  document.getElementById("autoBtn").addEventListener("click", function () {
    if (state.auto) stopAuto(); else startAuto();
  });
  document.getElementById("resetBtn").addEventListener("click", function () {
    resetDraft();
    render();
  });
  document.getElementById("ovBody").addEventListener("click", function (e) {
    const draftBtn = e.target.closest("[data-draft]");
    const row = e.target.closest(".ba-row");
    if (draftBtn || row) {
      if (state.live) {
        state.toast = "Draft in the host room — this overlay never submits a pick.";
        if (state.tab !== "roster") { /* stay on board */ }
        const recPn = nextMine(state.current);
        state.toast = "Take this player in the host draft (your next pick is " + (recPn ? pickLabel(recPn) : "done") + "). The overlay never submits.";
        render();
        return;
      }
      draftToMe((draftBtn || row).getAttribute(draftBtn ? "data-draft" : "data-id"));
      render();
      return;
    }
    const pos = e.target.closest("[data-pos]");
    if (pos) { state.pos = pos.getAttribute("data-pos"); render(); return; }
    if (e.target.closest("#sortBtn")) {
      state.sort = SORTS[(SORTS.indexOf(state.sort) + 1) % SORTS.length];
      render();
      return;
    }
    const link = e.target.closest("[data-link]");
    if (link) {
      const kind = link.getAttribute("data-link");
      state.toast = kind === "room"
        ? "Opens this live draft in BR Fantasy Draft Room — overlay stays synced, never submits."
        : "Opens your Draft Room cheat sheet for this league.";
      render();
      return;
    }
    const leg = e.target.closest("[data-legslot]");
    if (leg) {
      const slot = +leg.getAttribute("data-legslot");
      state.expanded = state.expanded === slot ? null : slot;
      render();
    }
  });

  clockTimer = setInterval(function () {
    if (state.live || draftDone() || state.auto) return;
    state.clock -= 1;
    if (state.clock <= 0) {
      simulateOne();
      state.clock = CLOCK_START;
      render();
      return;
    }
    const m = Math.floor(state.clock / 60), s = state.clock % 60;
    const el = document.getElementById("otcClock");
    if (el) el.textContent = m + ":" + String(s).padStart(2, "0");
  }, 1000);

  function postToHost(type, extra) {
    if (!EMBEDDED) return;
    try { window.parent.postMessage(Object.assign({ __br: "br-da", type: type }, extra || {}), "*"); }
    catch (_e) { /* ignore */ }
  }

  function fillSlotSel() {
    const sel = document.getElementById("slotSel");
    if (!sel) return;
    sel.innerHTML = "";
    for (let i = 1; i <= state.teams; i++) {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = i === state.mySlot ? i + " (you)" : String(i);
      if (i === state.mySlot) o.selected = true;
      sel.appendChild(o);
    }
  }

  function fillAdpSel() {
    const sel = document.getElementById("adpSel");
    if (!sel) return;
    const opts = (state.adpOptions && state.adpOptions.length) ? state.adpOptions : [];
    if (!opts.length) return;
    const want = state.adpSource || "consensus";
    sel.innerHTML = opts.map(function (o) {
      const v = String(o.value || o);
      const l = String(o.label || o);
      return '<option value="' + esc(v) + '"' + (v === want ? " selected" : "") + ">" + esc(l) + "</option>";
    }).join("");
    if (![].some.call(sel.options, function (o) { return o.value === want; }) && sel.options[0]) {
      sel.value = sel.options[0].value;
      state.adpSource = sel.value;
    } else {
      sel.value = want;
    }
  }

  function ingestPool(detail) {
    const rows = (detail && detail.players) || [];
    if (!rows.length) return;
    if (detail.adpSource && state.adpSource && String(detail.adpSource) !== String(state.adpSource)) return;
    if (Array.isArray(detail.adpOptions) && detail.adpOptions.length) state.adpOptions = detail.adpOptions;
    if (detail.adpSource) state.adpSource = String(detail.adpSource);
    fillAdpSel();
    players = rows.map(function (p) {
      return {
        id: String(p.id),
        name: p.name,
        pos: p.pos || "RB",
        team: p.team || "FA",
        adp: Number(p.adp) || 999,
        val: Number(p.val) || 0,
        ppg: p.ppg == null ? 0 : Number(p.ppg),
        age: p.age == null ? 0 : Number(p.age),
        bye: p.bye == null ? 0 : Number(p.bye),
        headshot: p.headshot || "",
        tier: p.tier || 6
      };
    });
    byId = {};
    byName = {};
    players.forEach(function (p) {
      if (p.id) byId[String(p.id)] = p;
      byName[normName(p.name)] = p;
    });
    state.sitePool = true;
    let maxVal = 1;
    players.forEach(function (p) { if (p.val > maxVal) maxVal = p.val; });
    state.valCap = maxVal;
    if (lastLiveDetail) ingestLive(lastLiveDetail);
    else render();
  }

  function matchLivePlayer(raw) {
    const pid = raw && raw.playerId != null && String(raw.playerId) !== "" ? String(raw.playerId) : "";
    if (pid && byId[pid]) return byId[pid];
    const name = (raw && (raw.playerName || raw.name)) || "";
    const key = normName(name);
    if (key && byName[key]) return byName[key];
    const pos = String((raw && (raw.pos || raw.position)) || "WR").toUpperCase();
    const nfl = String((raw && (raw.nflTeam || raw.team || raw.proTeam)) || "").toUpperCase().slice(0, 3) || "FA";
    const pn = Number((raw && (raw.overallPickNumber || raw.pick_no)) || 0) || state.current;
    const stub = {
      id: "live-" + (pid || key || pn),
      name: name || ("Pick " + pn),
      pos: POS[pos] ? pos : "WR",
      team: nfl,
      bye: BYE[nfl] || 10,
      adp: 999,
      tier: 6,
      ppg: 0,
      age: 0,
      val: 0,
      headshot: ""
    };
    if (!EMBEDDED && !state.sitePool && !byId[stub.id]) {
      players.push(stub);
      byId[stub.id] = stub;
      byName[normName(stub.name)] = stub;
    }
    return stub;
  }

  function ingestLive(detail) {
    if (!detail) return;
    lastLiveDetail = detail;
    state.live = true;
    stopAuto();
    if (detail.platform) state.platform = String(detail.platform).toLowerCase();
    if (detail.teams) state.teams = Math.max(2, Number(detail.teams) || state.teams);
    if (detail.rounds) state.rounds = Math.max(1, Number(detail.rounds) || state.rounds);
    if (detail.mySlot) state.mySlot = Math.max(1, Math.min(state.teams, Number(detail.mySlot)));
    if (detail.teamNames && typeof detail.teamNames === "object") state.teamNames = detail.teamNames;
    const raw = Array.isArray(detail.picks) ? detail.picks.slice() : [];
    raw.sort(function (a, b) {
      return (Number(a.overallPickNumber || a.pick_no || 0) - Number(b.overallPickNumber || b.pick_no || 0));
    });
    state.picks = [];
    state.drafted = {};
    raw.forEach(function (rp) {
      const pn = Number(rp.overallPickNumber || rp.pick_no || 0);
      if (!pn) return;
      const p = matchLivePlayer(rp);
      const slot = Number(rp.slot || rp.draftSlot || rp.draft_slot || rp.teamId || ownerOf(pn));
      const counts = posCounts(teamPicks(slot));
      const need = needOf(counts, p.pos) > 0;
      const grade = pickLetter(pn - p.adp, need);
      const ps = pickScore(p, counts, pn);
      state.picks.push({ pn: pn, slot: slot, p: p, grade: grade, ps: ps });
      state.drafted[p.id] = true;
    });
    state.current = raw.length ? (Number(raw[raw.length - 1].overallPickNumber || raw.length) + 1) : 1;
    state.clock = CLOCK_START;
    if (detail.syncText) {
      const chip = document.getElementById("syncChip");
      if (chip) chip.innerHTML = "<i></i> " + esc(detail.syncText);
    }
    fillSlotSel();
    render();
  }

  function setSyncStatus(ok, text) {
    const chip = document.getElementById("syncChip");
    if (!chip) return;
    chip.innerHTML = "<i></i> " + esc(text || (ok ? "SYNCED" : "Waiting"));
    chip.style.color = ok ? "" : "var(--warn)";
  }

  function setCollapsedUi(on) {
    const btn = document.getElementById("collapseBtn");
    if (!btn) return;
    const collapsed = !!on;
    document.documentElement.classList.toggle("br-da-rail", collapsed);
    btn.title = collapsed ? "Open overlay" : "Collapse overlay";
    btn.setAttribute("aria-label", collapsed ? "Open overlay" : "Collapse overlay");
    const path = btn.querySelector("path");
    if (path) path.setAttribute("d", collapsed ? "M15 6l-6 6 6 6" : "M9 6l6 6-6 6");
  }

  const recBtn = document.getElementById("reconnectBtn");
  if (recBtn) recBtn.addEventListener("click", function () { postToHost("reconnect"); });
  const colBtn = document.getElementById("collapseBtn");
  if (colBtn) colBtn.addEventListener("click", function () { postToHost("collapse"); });
  const slotSel = document.getElementById("slotSel");
  if (slotSel) slotSel.addEventListener("change", function () {
    state.mySlot = Number(slotSel.value) || state.mySlot;
    try { localStorage.setItem("br-da-slot", String(state.mySlot)); } catch (_e) { /* ignore */ }
    postToHost("slot", { mySlot: state.mySlot });
    fillSlotSel();
    render();
  });
  const adpSel = document.getElementById("adpSel");
  if (adpSel) adpSel.addEventListener("change", function () {
    const next = String(adpSel.value || "consensus");
    if (next === state.adpSource) return;
    state.adpSource = next;
    try { localStorage.setItem("br-da-adp", state.adpSource); } catch (_e) { /* ignore */ }
    if (EMBEDDED) {
      state.sitePool = false;
      render();
      postToHost("adp", { adpSource: state.adpSource });
    }
  });

  window.addEventListener("message", function (ev) {
    const msg = ev.data;
    if (!msg || msg.__br !== "br-da") return;
    if (msg.type === "pool") ingestPool(msg);
    if (msg.type === "picks") ingestLive(msg);
    if (msg.type === "sync") setSyncStatus(!!msg.ok, msg.text);
    if (msg.type === "collapsed") setCollapsedUi(msg.on);
    if (msg.type === "theme" && msg.theme) applyTheme(msg.theme);
  });

  document.addEventListener("error", function (e) {
    const t = e.target;
    if (!t || t.tagName !== "IMG") return;
    const fb = t.getAttribute("data-fallback");
    if (fb) {
      t.removeAttribute("data-fallback");
      t.src = fb;
      return;
    }
    const wrap = t.closest(".has-photo");
    if (wrap) wrap.classList.remove("has-photo");
    t.remove();
  }, true);

  if (!EMBEDDED) {
    buildPool();
    indexNames();
  }
  const savedSlot = Number(localStorage.getItem("br-da-slot") || 0);
  if (savedSlot) state.mySlot = savedSlot;
  fillSlotSel();
  try {
    const savedAdp = localStorage.getItem("br-da-adp");
    if (savedAdp) state.adpSource = savedAdp;
  } catch (_e) { /* ignore */ }
  fillAdpSel();
  const saved = localStorage.getItem("br-da-theme") || "system";
  applyTheme(saved);
  render();
  postToHost("ready", { adpSource: state.adpSource, mySlot: state.mySlot });
})();
