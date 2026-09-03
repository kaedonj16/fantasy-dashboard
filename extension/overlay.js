(function () {
  const CLOCK_START = 75;
  const SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"];
  const POS = { QB: "#3b82f6", RB: "#22c55e", WR: "#f59e0b", TE: "#8b5cf6", FLEX: "#14b8a6", SF: "#0ea5e9", K: "#64748b", DEF: "#475569", BN: "#64748b" };
  const SORTS = ["rec", "ps", "val", "proj", "adp"];
  const SORT_LBL = { rec: "Recommendation Rank", adp: "ADP", ps: "Pick Score", proj: "Proj PPG", val: "Value" };
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
    mySlot: EMBEDDED ? 0 : 7,
    slotAuto: false,
    live: EMBEDDED,
    hostInProgress: null,
    hostDrafted: null,
    teamNames: {},
    pickOwners: {},
    slotToRosterId: {},
    leagueId: "",
    season: 0,
    draftType: "redraft",
    leagueKind: "",
    formatLabel: "",
    orderFormat: "",
    orderLabel: "",
    bestBall: false,
    hostClock: null,
    hostClockAt: 0,
    pickTimer: 0,
    current: 1,
    picks: [],
    drafted: {},
    auto: false,
    tab: "board",
    pos: "ALL",
    sort: "rec",
    query: "",
    platform: "sleeper",
    clock: CLOCK_START,
    expanded: null,
    toast: "",
    syncOk: true,
    valCap: 180,
    sitePool: false,
    sf: false,
    leagueName: "",
    roster: null,
    ppr: 1,
    tep: 0,
    passTd: 4,
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
  let lastLiveFp = "";
  let availCache = null;
  let cliffLeft = null;
  let renderQueued = false;
  let compareIds = [];
  let summaryShown = false;
  let summaryOpen = false;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }
  function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
  function ownerOf(pn) {
    const mapped = Number(state.pickOwners && state.pickOwners[pn]);
    if (mapped >= 1 && mapped <= state.teams) return mapped;
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
  function draftDone() {
    if (state.live) {
      if (state.hostInProgress === true) return false;
      if (state.hostDrafted === true) return true;
      return false;
    }
    return state.current > state.teams * state.rounds;
  }
  function available() {
    if (availCache) return availCache;
    availCache = players.filter(function (p) { return !state.drafted[p.id]; });
    return availCache;
  }
  function liveFingerprint(detail) {
    const raw = (detail && detail.picks) || [];
    const last = raw.length ? raw[raw.length - 1] : null;
    return [
      raw.length,
      last ? (last.overallPickNumber || last.pick_no || 0) : 0,
      last ? (last.playerId || "") : "",
      last ? (last.playerName || last.name || "") : "",
      detail && detail.mySlot || "",
      detail && detail.teams || "",
      detail && detail.rounds || "",
      detail && detail.inProgress ? 1 : 0,
      detail && detail.drafted ? 1 : 0,
      detail && detail.platform || "",
      detail && detail.sf ? 1 : 0,
      detail && detail.leagueName || "",
      detail && detail.ppr != null ? detail.ppr : "",
      detail && detail.tep != null ? detail.tep : "",
      detail && detail.passTd != null ? detail.passTd : "",
      detail && detail.teamNames ? Object.keys(detail.teamNames).length : "",
      detail && detail.pickOwners ? Object.keys(detail.pickOwners).length : "",
      window.BRDraftSlot && BRDraftSlot.rosterKey ? BRDraftSlot.rosterKey(detail && detail.roster) : ""
    ].join("|");
  }
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
  function normPos(pos) {
    if (window.BRDraftSlot && BRDraftSlot.normDraftPos) return BRDraftSlot.normDraftPos(pos);
    const p = String(pos || "").toUpperCase();
    if (p === "PK") return "K";
    if (p === "DST" || p === "D/ST" || p === "D-ST" || p === "D ST") return "DEF";
    return p;
  }
  function isKDef(p) {
    if (window.BROverlayScore && BROverlayScore.isKDef) return BROverlayScore.isKDef(p);
    const pos = normPos(p && (p.pos || p.position || p));
    return pos === "K" || pos === "DEF";
  }
  function posCounts(list) {
    const c = { QB: 0, RB: 0, WR: 0, TE: 0, K: 0, DEF: 0 };
    list.forEach(function (p) {
      const pos = normPos(p.pos || p.position);
      if (c[pos] != null) c[pos]++;
    });
    return c;
  }
  function posTargets() {
    const rs = state.roster;
    if (rs) {
      return {
        QB: (rs.QB || 0) + (rs.SF || 0) || (state.sf ? 2 : 1),
        RB: (rs.RB || 0) + Math.min(1, rs.FLEX || 0) || 3,
        WR: (rs.WR || 0) + Math.min(1, rs.FLEX || 0) || 3,
        TE: (rs.TE || 0) || 1,
      };
    }
    return { QB: state.sf ? 2 : 1, RB: 3, WR: 3, TE: 1 };
  }
  function needOf(counts, pos) {
    const t = posTargets();
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

  function scoreCtx() {
    return {
      current: state.current,
      teams: state.teams,
      rounds: state.rounds,
      mySlot: state.mySlot,
      pickOwners: state.pickOwners,
      sf: !!state.sf,
      type: (window.BRDraftSlot && BRDraftSlot.normDraftType)
        ? BRDraftSlot.normDraftType(state.draftType)
        : (state.draftType === "rookie" ? "rookie"
          : (state.draftType === "startup" || state.draftType === "dynasty" ? "startup" : "redraft")),
      tep: Number(state.tep) || 0,
      ppr: state.ppr != null ? Number(state.ppr) : 1,
      passTd: state.passTd >= 6 ? 6 : 4,
      roster: state.roster || undefined,
      picks: state.picks,
    };
  }
  function pickScore(p, counts, pickNo) {
    if (isKDef(p)) return null;
    if (p && p._ps != null) return p._ps;
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
    if (isKDef(p)) return null;
    if (p && p._ds != null) return p._ds;
    const ps = pickScore(p, counts, pickNo);
    if (ps == null) return null;
    const need = needOf(counts, p.pos);
    let ds = ps + need * 8 + (p.tier <= 2 ? 4 : 0);
    if (!need && p.pos === "QB" && (counts.QB || 0) >= 1) ds -= 18;
    if (!need && p.pos === "TE" && (counts.TE || 0) >= 1) ds -= 8;
    return ds;
  }
  function rankedPool(counts, pickNo) {
    if (window.BROverlayScore && window.BRPickScore && window.DraftBoardCore && state.sitePool) {
      try {
        return BROverlayScore.rankPool(players, available(), scoreCtx());
      } catch (_e) { /* fall through to local ranker */ }
    }
    const pool = available().filter(function (p) { return !isKDef(p); });
    for (let i = 0; i < pool.length; i++) {
      const p = pool[i];
      p._ps = pickScore(p, counts, pickNo);
      p._ds = decisionScore(p, counts, pickNo);
    }
    pool.sort(function (a, b) { return (b._ds - a._ds) || (b._ps - a._ps) || (a.adp - b.adp); });
    for (let i = 0; i < pool.length; i++) pool[i]._rank = i + 1;
    return pool;
  }
  function isTierCliff(p) {
    if (!cliffLeft) {
      cliffLeft = {};
      const pool = available();
      for (let i = 0; i < pool.length; i++) {
        const q = pool[i];
        const k = q.pos + ":" + q.tier;
        cliffLeft[k] = (cliffLeft[k] || 0) + 1;
      }
    }
    const left = cliffLeft[p.pos + ":" + p.tier] || 0;
    return p.tier <= 2 && left <= 2 && state.current > state.teams;
  }
  function reasonsFor(p, counts, pickNo) {
    const out = [];
    const need = needOf(counts, p.pos);
    const diff = Math.round(pickNo - p.adp);
    if (need > 0) out.push("Fills a starting " + p.pos + " need (" + need + " still open)");
    else if (p.pos !== "QB" && (counts.RB + counts.WR + counts.TE) < 5) out.push("FLEX-eligible depth for weekly lineup");
    if (diff >= 4) out.push("Value vs ADP: " + diff + " picks past market");
    else if (diff <= -4) out.push("Slight reach vs ADP (" + Math.abs(diff) + " early) - still a positional fit");
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
    summaryShown = false;
    summaryOpen = false;
    stopAuto();
  }

  function slotList() {
    if (window.BROverlayScore && BROverlayScore.optimalLineup) {
      const ol = BROverlayScore.optimalLineup([], scoreCtx());
      if (ol && ol.starters && ol.starters.length >= 4) {
        return ol.starters.map(function (s) { return s.slot; });
      }
    }
    if (window.BRDraftSlot && BRDraftSlot.slotListFromRoster && state.roster) {
      const list = BRDraftSlot.slotListFromRoster(state.roster);
      if (list && list.length >= 4) return list;
    }
    return SLOTS.slice();
  }
  function slotEligible(slot, pos) {
    const p = String(pos || "").toUpperCase();
    const s = String(slot || "").toUpperCase();
    if (s === "FLEX") return p === "RB" || p === "WR" || p === "TE";
    if (s === "SF" || s === "OP") return p === "QB" || p === "RB" || p === "WR" || p === "TE";
    if (s === "RB_WR") return p === "RB" || p === "WR";
    if (s === "WR_TE") return p === "WR" || p === "TE";
    if (s === "RB_TE") return p === "RB" || p === "TE";
    if (s === "DEF") return normPos(p) === "DEF";
    return normPos(p) === s;
  }
  function optimalLineup(list) {
    if (window.BROverlayScore && BROverlayScore.optimalLineup) {
      return BROverlayScore.optimalLineup(list, scoreCtx());
    }
    const leftover = list.slice().sort(function (a, b) { return (b.ppg || 0) - (a.ppg || 0); });
    const starters = [];
    function take(slot, ok) {
      const i = leftover.findIndex(ok);
      if (i < 0) { starters.push({ slot: slot, p: null }); return; }
      starters.push({ slot: slot, p: leftover.splice(i, 1)[0] });
    }
    slotList().forEach(function (slot) {
      take(slot, function (p) { return slotEligible(slot, p.pos); });
    });
    return { starters: starters, bench: leftover };
  }

  function competitiveWindow(list) {
    if (state.draftType === "redraft") return null;
    const ol = optimalLineup(list);
    let wSum = 0, aSum = 0;
    ol.starters.forEach(function (x) {
      if (!x.p) return;
      const w = Math.max(1, x.p.val || 1);
      aSum += x.p.age * w; wSum += w;
    });
    if (wSum <= 0) return null;
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
    const coverage = filled / Math.max(1, slotList().length);
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

  function picksBySlot() {
    const out = {};
    for (let s = 1; s <= state.teams; s++) out[s] = [];
    state.picks.forEach(function (x) {
      const slot = Number(x.slot) || 0;
      if (!out[slot]) out[slot] = [];
      out[slot].push({ pn: x.pn, p: x.p });
    });
    Object.keys(out).forEach(function (k) {
      out[k].sort(function (a, b) { return a.pn - b.pn; });
    });
    return out;
  }

  function gradeAllTeams() {
    const bySlot = picksBySlot();
    if (window.BROverlayScore && BROverlayScore.gradeField && (state.sitePool || !EMBEDDED)) {
      try {
        const field = BROverlayScore.gradeField(players, bySlot, scoreCtx());
        if (field && field.length) {
          return field.map(function (t) {
            t.name = t.isMe ? "You" : teamName(t.slot);
            t.picks = teamPicks(t.slot);
            return t;
          });
        }
      } catch (_e) { /* fall through only for the standalone mockup */ }
    }
    if (EMBEDDED) return [];
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
        name: (i + 1 === state.mySlot) ? "You" : teamName(i + 1),
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

  function teamStrengthPPG(list) {
    const ol = optimalLineup(list);
    let s = 0;
    ol.starters.forEach(function (x) {
      if (!x.p) return;
      const v = (window.BROverlayScore && BROverlayScore.lineupScore)
        ? BROverlayScore.lineupScore(x.p, scoreCtx())
        : (Number(x.p.ppg) || 0);
      if (isFinite(v) && v > 0 && v !== -Infinity) s += v;
    });
    return s;
  }

  let poMcCache = null, poMcSig = null;
  function playoffOddsBySlot(all) {
    const sig = all.map(function (t) { return t.slot + ":" + (t.picks ? t.picks.length : 0); }).join("|") + "@" + state.current;
    if (poMcCache && poMcSig === sig) return poMcCache;
    const teams = all.map(function (t) {
      return { slot: t.slot, S: teamStrengthPPG((t.picks || []).map(function (x) { return x.p || x; }).filter(Boolean)) };
    });
    const n = teams.length;
    const odds = {};
    teams.forEach(function (t) { odds[t.slot] = 0; });
    if (n >= 2) {
      let spots = n <= 8 ? 4 : 6;
      if (spots >= n) spots = Math.max(1, n - 1);
      const W = 14, N = 2500, sigma = 27;
      function gauss() {
        let u = 0, v = 0;
        while (!u) u = Math.random();
        while (!v) v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      for (let s = 0; s < N; s++) {
        const wins = [], pts = [];
        for (let t = 0; t < n; t++) { wins[t] = 0; pts[t] = 0; }
        for (let w = 0; w < W; w++) {
          const idx = [];
          for (let q = 0; q < n; q++) idx[q] = q;
          for (let i = idx.length - 1; i > 0; i--) {
            const j = (Math.random() * (i + 1)) | 0;
            const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
          }
          for (let k = 0; k + 1 < idx.length; k += 2) {
            const a = idx[k], b = idx[k + 1];
            const sa = teams[a].S + gauss() * sigma;
            const sb = teams[b].S + gauss() * sigma;
            pts[a] += sa; pts[b] += sb;
            if (sa >= sb) wins[a]++; else wins[b]++;
          }
        }
        const ord = [];
        for (let o = 0; o < n; o++) ord[o] = o;
        ord.sort(function (x, y) { return (wins[y] - wins[x]) || (pts[y] - pts[x]); });
        for (let r = 0; r < spots; r++) odds[teams[ord[r]].slot] += 1;
      }
      teams.forEach(function (t) { odds[t.slot] = Math.round(odds[t.slot] / N * 100); });
    }
    poMcCache = odds; poMcSig = sig;
    return odds;
  }

  let poServer = null, poServerSig = null, poFetching = false, poFailedSig = null;
  function poFmt(po) {
    const n = Number(po);
    if (!isFinite(n)) return "";
    if (n >= 100) return "100";
    if (n <= 0) return "0";
    return n.toFixed(1);
  }
  function poColor(po) {
    return po >= 60 ? "#22c55e" : po >= 35 ? "#f59e0b" : "#ef4444";
  }
  function refreshServerPlayoffOdds(all) {
    if (!draftDone() || !all || all.length < 2) return;
    const sig = all.map(function (t) { return t.slot + ":" + (t.picks ? t.picks.length : 0); }).join("|") + "@" + state.current;
    if (poFetching || (poServer && poServerSig === sig) || poFailedSig === sig) return;
    if (!state.leagueId || typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.sendMessage) {
      poFailedSig = sig;
      return;
    }
    poFetching = true;
    const liveLeague = !!(state.leagueId && state.platform);
    const teamsPayload = all.map(function (t) {
      const rid = state.slotToRosterId[t.slot] || state.slotToRosterId[String(t.slot)] || t.slot;
      return {
        slot: t.slot,
        roster_id: rid,
        name: t.name,
        players: liveLeague ? [] : (t.picks || []).map(function (x) {
          return (x.p && x.p.id != null) ? String(x.p.id) : (x.id != null ? String(x.id) : null);
        }).filter(Boolean),
      };
    });
    chrome.runtime.sendMessage({
      type: "fetchDraftPlayoffOdds",
      season: state.season || 0,
      ppr: state.ppr,
      tep: state.tep,
      passTd: state.passTd,
      roster: state.roster,
      playoffTeams: state.teams <= 8 ? 4 : 6,
      platform: state.platform || "sleeper",
      leagueId: state.leagueId,
      useLeague: true,
      viewerSlot: state.mySlot || null,
      teams: teamsPayload,
    }, function (resp) {
      void chrome.runtime.lastError;
      poFetching = false;
      if (resp && resp.odds && resp.odds.length) {
        const m = {};
        resp.odds.forEach(function (o) { if (o.slot != null) m[o.slot] = o.playoff_pct; });
        poServer = m; poServerSig = sig; poFailedSig = null;
        render();
      } else {
        poFailedSig = sig;
        render();
      }
    });
  }

  function playoffOdds(all) {
    if (!draftDone()) return {};
    refreshServerPlayoffOdds(all);
    const sig = all.map(function (t) { return t.slot + ":" + (t.picks ? t.picks.length : 0); }).join("|") + "@" + state.current;
    if (poServer && poServerSig === sig) return poServer;
    if (poFailedSig === sig) return playoffOddsBySlot(all);
    return {};
  }

  function playoffOddsPending(all) {
    if (!draftDone() || !all || all.length < 2) return false;
    const sig = all.map(function (t) { return t.slot + ":" + (t.picks ? t.picks.length : 0); }).join("|") + "@" + state.current;
    return !(poServer && poServerSig === sig) && poFailedSig !== sig;
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
    if (!isFinite(a) || a >= 900) return "-";
    return a.toFixed(1);
  }
  function fmtPpg(p) {
    const n = Number(p && p.ppg);
    if (!isFinite(n) || n <= 0) return "-";
    return n.toFixed(1);
  }
  function hsUrl(p) {
    if (!p) return "";
    if (p.headshot) return String(p.headshot);
    const id = String(p.id || "");
    if (/^\d+$/.test(id)) return "https://sleepercdn.com/content/nfl/players/" + id + ".jpg";
    return "";
  }
  function hsMark(p, cls, opts) {
    const pc = POS[p.pos] || POS.WR;
    const url = hsUrl(p);
    if (!url) return '<span class="' + cls + '" style="--pc:' + pc + '" aria-hidden="true"></span>';
    const fallback = /^\d+$/.test(String(p.id || ""))
      ? "https://sleepercdn.com/content/nfl/players/" + p.id + ".jpg"
      : "";
    const extra = fallback && fallback !== url ? ' data-fallback="' + esc(fallback) + '"' : "";
    const eager = opts && opts.eager;
    return '<span class="' + cls + ' has-photo" style="--pc:' + pc + '" aria-hidden="true"><img alt="" src="' + esc(url) + '"' + extra + (eager ? ' fetchpriority="high"' : ' loading="lazy"') + ' decoding="async"></span>';
  }

  function playerRow(p, opts) {
    opts = opts || {};
    const cliff = isTierCliff(p);
    const pc = POS[p.pos];
    const mine = myPicks();
    const byeLvl = !isKDef(p) && mine.filter(function (x) { return !isKDef(x) && x.bye === p.bye; }).length >= 2;
    let chip;
    if (isKDef(p) && p._rank == null) {
      chip = "";
    } else if (state.sort === "rec") {
      chip = '<div class="pschip recchip" title="Recommendation rank">#' + (opts.rank || p._rank) + "<small>REC</small></div>";
    } else if (state.sort === "ps" && p._ps != null) {
      const shown = p._psShow != null ? p._psShow : p._ps;
      const col = psColor(shown);
      chip = '<div class="pschip" style="color:' + col + ";background:" + col + '1a" title="Pick Score">'+ shown + "<small>PS</small></div>";
    } else {
      chip = "";
    }
    const reason = opts.reason ? '<div class="ba-reason">' + esc(opts.reason) + "</div>" : "";
    const onCmp = compareIds.indexOf(String(p.id)) >= 0;
    return '<div class="ba-row' + (onCmp ? " is-cmp" : "") + '" data-id="' + p.id + '">'
      + hsMark(p, "hs", { eager: (opts.rank || 99) <= 6 })
      + '<div class="ba-body"><div class="ba-name">' + esc(p.name) + "</div>"
      + '<div class="ba-meta"><span class="posb" style="background:' + pc + '">' + p.pos + "</span>"
      + esc(p.team)
      + '<span class="tier' + (cliff ? " cliff" : "") + '">T' + p.tier + "</span>"
      + '<span class="tabular">' + fmtPpg(p) + " proj</span>"
      + (byeLvl ? '<span class="bye-flag">Bye ' + p.bye + "</span>" : "")
      + "</div>" + reason + "</div>"
      + '<div class="ba-right"><div class="ba-val">' + p.val + '</div><div class="ba-sub">ADP ' + fmtAdp(p) + "</div></div>"
      + chip
      + '<button type="button" class="dr-cmp-btn' + (onCmp ? " on" : "") + '" data-cmp="' + esc(String(p.id)) + '" title="Compare" aria-label="Compare ' + esc(p.name) + '" aria-pressed="' + (onCmp ? "true" : "false") + '">vs</button>'
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

  function playoffOddsFor(all, slot) {
    const odds = playoffOdds(all)[slot];
    return odds != null && isFinite(Number(odds)) ? Number(odds) : null;
  }

  function finalGradeCard(all) {
    const me = (all || []).filter(function (t) { return t.isMe; })[0] || (all || [])[0];
    if (!me) return "";
    const rank = all.findIndex(function (t) { return t.isMe; }) + 1;
    const shownRank = rank >= 1 ? rank : 1;
    const odds = playoffOddsFor(all, me.slot);
    const oddsHtml = odds != null
      ? ('<b class="tabular" style="color:' + (odds >= 50 ? "var(--win)" : "var(--warn)") + '">' + odds.toFixed(1) + "%</b>")
      : "<b>-</b>";
    return '<button type="button" class="rec-card final-grade" data-open-summary="1" title="Open draft summary">'
      + '<div class="rec-label">Draft Report Card</div>'
      + '<div class="letter" style="color:' + gradeCol(me.grade.score) + '">' + gradeLetter(me.grade.score) + "</div>"
      + '<div class="rank">League rank #' + shownRank + " of " + state.teams + "</div>"
      + '<div class="sub">Projected playoff odds ' + oddsHtml + "</div>"
      + '<div class="sum-open">Open summary</div></button>';
  }

  function maybeShowSummary() {
    if (summaryShown || !draftDone() || !state.picks.length) return;
    summaryShown = true;
    state.tab = "grades";
    summaryOpen = true;
  }

  function openSummary() {
    if (!draftDone() && !myPicks().length) return;
    summaryOpen = true;
    render();
  }

  function closeSummary() {
    summaryOpen = false;
    const modal = document.getElementById("sumModal");
    if (modal) modal.hidden = true;
  }

  function summaryRow(slot, p) {
    if (!p) {
      return '<div class="dr-sum-row"><span class="dr-sum-slot" style="background:' + slotColor(slot) + '">' + slot
        + '</span><span class="dr-sum-empty">open</span></div>';
    }
    const pn = p.pn || 0;
    const pickStr = pn ? pickLabel(pn) : "";
    const ps = p._psShow != null ? p._psShow : (p.ps != null ? Math.round(p.ps) : null);
    const psStr = ps != null
      ? '<span class="dr-sum-ps" style="color:' + psColor(ps) + '">' + ps + "</span>"
      : "";
    return '<div class="dr-sum-row">'
      + '<span class="dr-sum-slot" style="background:' + slotColor(slot) + '">' + slot + "</span>"
      + '<div class="dr-sum-body"><div class="dr-sum-name">' + esc(p.name) + "</div>"
      + '<div class="dr-sum-meta">' + esc(p.pos || "") + (p.team ? " · " + esc(p.team) : "")
      + (pickStr ? " · " + pickStr : "") + "</div></div>"
      + psStr + "</div>";
  }

  function summaryHtml() {
    const all = gradeAllTeams();
    const me = (all || []).filter(function (t) { return t.isMe; })[0] || (all || [])[0];
    if (!me) return "";
    const g = me.grade;
    const gMax = gradeMax();
    const ol = optimalLineup(me.picks);
    const arch = me.archetype || (window.BROverlayScore && BROverlayScore.teamArchetype
      ? BROverlayScore.teamArchetype(picksBySlot()[state.mySlot] || [], scoreCtx())
      : null);
    const odds = playoffOddsFor(all, me.slot);
    const oddsPending = draftDone() && playoffOddsPending(all);
    let stats = [];
    const starters = (ol.starters || []).filter(function (s) { return s.p; });
    let proj = 0;
    let projN = 0;
    starters.forEach(function (s) {
      const v = (window.BROverlayScore && BROverlayScore.ppgOf)
        ? BROverlayScore.ppgOf(s.p, scoreCtx())
        : (Number(s.p.ppg) || 0);
      if (v != null && isFinite(v) && v > 0) { proj += v; projN++; }
    });
    if (projN >= 2) stats.push({ v: proj.toFixed(1), l: "Proj PPG" });
    if (oddsPending) stats.push({ v: "…", l: "Playoff Odds" });
    else if (odds != null) stats.push({ v: poFmt(odds) + "%", l: "Playoff Odds" });
    const profile = arch && arch.label
      ? (arch.label + (g.window && g.window.label ? " · " + g.window.label : ""))
      : (g.window && g.window.label ? g.window.label : "");
    let html = '<button type="button" class="dr-cmp-close" id="drSumClose" data-sum-close="1" aria-label="Close">&times;</button>'
      + '<div class="dr-sum-title" id="sumTitle">Draft Report Card</div>'
      + '<div class="dr-sum-grade-wrap"><div class="dr-sum-letter" style="color:' + gradeCol(g.score) + '">' + gradeLetter(g.score) + "</div>"
      + '<div class="dr-sum-bars">'
      + (g.provisional ? '<div class="grade-early">Early - still forming</div>' : "")
      + gbar("Value", g.value, gMax.value) + gbar("Starters", g.starters, gMax.starters)
      + gbar("Construction", g.construction, gMax.construction)
      + "</div></div>";
    if (stats.length) {
      html += '<div class="dr-sum-stats">';
      stats.forEach(function (s) {
        html += '<div class="dr-sum-stat"><div class="dr-sum-stat-v">' + s.v + '</div><div class="dr-sum-stat-l">' + s.l + "</div></div>";
      });
      html += "</div>";
    }
    if (profile) html += '<div class="dr-sum-arch">' + esc(profile) + "</div>";
    html += '<div class="dr-sum-section">Starters</div>';
    (ol.starters || []).forEach(function (s) { html += summaryRow(s.slot, s.p); });
    html += '<div class="dr-sum-section">Bench</div>';
    if (ol.bench && ol.bench.length) ol.bench.forEach(function (p) { html += summaryRow("BN", p); });
    else html += summaryRow("BN", null);
    html += '<div class="dr-sum-foot"><button type="button" class="btn btn-ghost" data-sum-close="1">Close</button></div>';
    return html;
  }

  function paintSummary() {
    const modal = document.getElementById("sumModal");
    const card = document.getElementById("sumCard");
    if (!modal || !card) return;
    if (!summaryOpen) {
      modal.hidden = true;
      return;
    }
    card.innerHTML = summaryHtml();
    modal.hidden = false;
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
      html += finalGradeCard(gradeAllTeams());
    } else if (pool[0]) {
      html += bannersHtml(counts, pool[0]);
    }
    if (compareIds.length === 1) {
      const waiting = byId[compareIds[0]];
      html += '<div class="cmp-hint">Comparing ' + esc(waiting ? waiting.name : "player") + " - tap vs on another</div>";
    }
    const q = (state.query || "").trim().toLowerCase();
    const kdAvail = available().filter(isKDef).sort(function (a, b) {
      return (window.BROverlayScore && BROverlayScore.sortKdef)
        ? BROverlayScore.sortKdef(a, b)
        : ((a.adp || 999) - (b.adp || 999)) || ((b.ppg || 0) - (a.ppg || 0));
    });
    let promoted = [];
    if (state.pos === "ALL" && kdAvail.length && window.BROverlayScore && BROverlayScore.kdefNeed) {
      const need = BROverlayScore.kdefNeed(scoreCtx(), counts);
      if (need) {
        if (need.needK > 0) {
          const bk = kdAvail.filter(function (p) { return normPos(p.pos) === "K"; })[0];
          if (bk) promoted.push(bk);
        }
        if (need.needDef > 0) {
          const bd = kdAvail.filter(function (p) { return normPos(p.pos) === "DEF"; })[0];
          if (bd) promoted.push(bd);
        }
      }
    }
    const promotedIds = {};
    promoted.forEach(function (p) { promotedIds[String(p.id)] = true; });
    let kdRest = kdAvail.filter(function (p) { return !promotedIds[String(p.id)]; });
    let rows = pool.filter(function (p) { return !isKDef(p); });
    if (state.pos === "K" || state.pos === "DEF") {
      rows = kdAvail.filter(function (p) { return normPos(p.pos) === state.pos; });
      promoted = [];
      kdRest = [];
    } else if (state.pos !== "ALL") {
      rows = rows.filter(function (p) { return normPos(p.pos) === state.pos; });
      promoted = [];
      kdRest = [];
    }
    if (q) {
      const hit = function (p) { return String(p.name).toLowerCase().indexOf(q) >= 0; };
      rows = rows.filter(hit);
      promoted = promoted.filter(hit);
      kdRest = kdRest.filter(hit);
    }
    if (state.sort === "adp") rows = rows.slice().sort(function (a, b) { return a.adp - b.adp; });
    else if (state.sort === "ps") rows = rows.slice().sort(function (a, b) { return (b._ps || 0) - (a._ps || 0); });
    else if (state.sort === "proj") rows = rows.slice().sort(function (a, b) { return (b.ppg || 0) - (a.ppg || 0); });
    else if (state.sort === "val") rows = rows.slice().sort(function (a, b) { return (b.val || 0) - (a.val || 0); });
    promoted.forEach(function (p) {
      html += playerRow(p, { reason: "Fill your " + normPos(p.pos) + " slot before the draft ends" });
    });
    rows.slice(0, 40).forEach(function (p, i) {
      let reason = "";
      if (state.sort === "rec" && !isKDef(p)) {
        reason = (window.BROverlayScore && pool && pool._reasonCtx)
          ? (BROverlayScore.pickReason(p, pool) || "")
          : (reasonsFor(p, counts, recPn)[0] || "");
      }
      html += playerRow(p, { rank: state.sort === "rec" && p._rank != null ? p._rank : i + 1, reason: reason });
    });
    kdRest.forEach(function (p) { html += playerRow(p, {}); });
    if (!rows.length && !promoted.length && !kdRest.length) {
      html += '<div class="empty-log" style="color:var(--text-muted)">No players match this filter.</div>';
    }
    return html;
  }

  function gradeMax() {
    if (window.BROverlayScore && BROverlayScore.gradeMax) {
      return BROverlayScore.gradeMax(scoreCtx().type);
    }
    return { value: 20, starters: 50, construction: 30 };
  }
  function gbar(label, val, max) {
    const cap = max > 0 ? max : 100;
    const pct = Math.round(clamp((Number(val) || 0) / cap * 100, 0, 100));
    const col = pct >= 80 ? "#22c55e" : pct >= 60 ? "#38bdf8" : pct >= 40 ? "#f59e0b" : "#ef4444";
    return '<div class="gbar-row"><span class="gbar-lbl">' + label + "</span>"
      + '<div class="gbar"><div class="gbar-fill" style="width:' + pct + "%;background:" + col + '"></div></div>'
      + '<span class="gbar-pct" style="color:' + col + '">' + pct + "</span></div>";
  }

  function renderRoster() {
    const mine = myPicks();
    const all = gradeAllTeams();
    const me = all.filter(function (t) { return t.isMe; })[0] || all[0];
    if (!me) return '<div class="empty-log">Waiting on your draft seat...</div>';
    const g = me.grade;
    const settingsTxt = leagueSettingsLabel();
    let html = "";
    if (settingsTxt) html += '<div class="settings-line" id="leagueSettings">' + esc(settingsTxt) + "</div>";
    const gMax = gradeMax();
    const arch = me.archetype || (window.BROverlayScore && BROverlayScore.teamArchetype
      ? BROverlayScore.teamArchetype(picksBySlot()[state.mySlot] || [], scoreCtx())
      : null);
    html += '<div class="grade-card"><div><div class="grade-letter" style="color:' + gradeCol(g.score) + '">' + gradeLetter(g.score) + "</div>"
      + (g.provisional ? '<div class="grade-early">Early</div>' : "") + "</div>"
      + '<div class="grade-meta">'
      + (arch && arch.label ? '<div class="grade-pace">' + esc(arch.label) + "</div>" : "")
      + gbar("Value", g.value, gMax.value) + gbar("Starters", g.starters, gMax.starters) + gbar("Construction", g.construction, gMax.construction)
      + "</div></div>";

    const proj = (window.BROverlayScore && BROverlayScore.rosterProjection)
      ? BROverlayScore.rosterProjection(mine, players, scoreCtx())
      : null;
    if (proj) {
      const col = proj.pct >= 108 ? "#22c55e" : proj.pct >= 92 ? "#f59e0b" : "#ef4444";
      html += '<div class="proj-card"><div class="proj-title">Roster Projection</div><div class="proj-stats">'
        + '<div class="proj-stat"><div class="proj-val">' + proj.myAvg.toFixed(1) + '</div><div class="proj-lbl">My Avg PPG</div></div>'
        + (proj.lgAvg > 0 ? '<div class="proj-stat"><div class="proj-val">' + proj.lgAvg.toFixed(1) + '</div><div class="proj-lbl">Avg Player</div></div>' : "")
        + (proj.lgAvg > 0 ? '<div class="proj-stat"><div class="proj-val" style="color:' + col + '">' + proj.pct + '%</div><div class="proj-lbl">vs League</div></div>' : "")
        + '</div><div class="proj-bar"><div class="gbar-fill" style="width:' + Math.min(100, proj.pct) + "%;background:" + col + '"></div></div></div>';
    }
    const w = g.window;
    const wcls = w && w.label === "Future" ? "win-future" : w && w.label === "Win-Now" ? "win-winnow" : "win-balanced";
    html += '<div class="win-row">';
    if (w && w.label) html += '<span class="win-chip ' + wcls + '">' + esc(w.label) + "</span>";
    if (w && w.avgAge) html += '<span style="font-size:11px;color:var(--text-muted)">Avg age ' + w.avgAge.toFixed(1) + "</span>";
    if (draftDone()) {
      const odds = playoffOddsFor(all, me.slot);
      if (odds != null) {
        html += '<span class="odds-chip" style="color:' + (odds >= 50 ? "var(--win)" : "var(--warn)") + '">' + odds.toFixed(1) + "% playoff</span>";
      }
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
    return html;
  }

  function recapHtml(all) {
    const recap = (window.BROverlayScore && BROverlayScore.recapStats)
      ? BROverlayScore.recapStats(all)
      : null;
    if (!recap || !recap.steals.length) return "";
    function line(x) {
      const txt = recap.useGap && x.gap != null
        ? ((x.gap > 0 ? "+" : "") + x.gap)
        : String(Math.round(x.ps));
      const col = recap.useGap && x.gap != null
        ? (x.gap > 0 ? "var(--win)" : (x.gap < 0 ? "var(--loss)" : "var(--text-muted)"))
        : psColor(x.ps);
      return '<div class="recap-row"><span class="posb" style="background:' + (POS[x.pos] || POS.BN) + '">' + esc(x.pos || "-") + "</span>"
        + '<span class="recap-main"><span class="recap-name">' + esc(x.name) + "</span>"
        + '<span class="recap-sub">' + esc(x.team) + " · " + pickLabel(x.pn) + "</span></span>"
        + '<span class="recap-gap" style="color:' + col + '">' + txt + "</span></div>";
    }
    return '<div class="recap"><div><p class="recap-h">' + IC.gem + "Biggest steals</p>" + recap.steals.map(line).join("")
      + '</div><div><p class="recap-h">' + IC.down + "Biggest reaches</p>" + recap.reaches.map(line).join("")
      + "</div></div>"
      + '<p class="recap-h" style="padding:4px 12px 0">' + IC.bars + "By the numbers</p>"
      + '<div class="nums"><div class="tile"><div class="tile-l">Steal of the draft</div><div class="tile-b">' + esc(recap.steals[0].name) + '</div><div class="tile-s">' + esc(recap.steals[0].team) + " · " + pickLabel(recap.steals[0].pn) + "</div></div>"
      + '<div class="tile"><div class="tile-l">Biggest reach</div><div class="tile-b">' + esc(recap.reaches[0].name) + '</div><div class="tile-s">' + esc(recap.reaches[0].team) + " · " + pickLabel(recap.reaches[0].pn) + "</div></div>"
      + '<div class="tile"><div class="tile-l">Best value drafter</div><div class="tile-b">' + esc(recap.valueTeam) + '</div><div class="tile-s">Highest average pick grade</div></div>'
      + '<div class="tile"><div class="tile-l">Most drafted</div><div class="tile-b">' + recap.topPos + (recap.posCount[recap.topPos] ? " (" + recap.posCount[recap.topPos] + ")" : "") + '</div><div class="tile-s">' + recap.pickCount + " picks total</div></div></div>";
  }

  function renderGrades() {
    const all = gradeAllTeams();
    const odds = draftDone() ? playoffOdds(all) : {};
    const pending = draftDone() && playoffOddsPending(all);
    let html = draftDone() ? finalGradeCard(all) : "";
    html += recapHtml(all);
    html += '<p class="recap-h" style="padding:8px 12px 6px">' + IC.trophy + "Draft grades</p>";
    all.forEach(function (t, i) {
      const w = t.grade.window;
      const wcls = w && w.label === "Future" ? "win-future" : w && w.label === "Win-Now" ? "win-winnow" : "win-balanced";
      const open = state.expanded === t.slot;
      const winTag = w && w.label ? '<span class="win-chip ' + wcls + '">' + esc(w.label) + "</span>" : "";
      let poTag = "";
      if (draftDone()) {
        if (pending) poTag = '<span class="lpo" title="Calculating playoff odds">…</span>';
        else if (odds[t.slot] != null) {
          poTag = '<span class="lpo" style="color:' + poColor(odds[t.slot]) + '">' + poFmt(odds[t.slot]) + "%</span>";
        }
      }
      html += '<div class="lrow' + (t.isMe ? " is-me" : "") + (open ? " is-open" : "") + '" data-legslot="' + t.slot + '">'
        + rankMedal(i + 1)
        + '<span class="lname">' + esc(t.name) + "</span>"
        + winTag
        + poTag
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
    document.getElementById("syncChip").innerHTML = "<i aria-hidden=\"true\"></i><span class=\"sync-chip-txt\">" + esc(pf.sync) + "</span>";
    const rd = Math.min(state.rounds, Math.ceil(Math.min(state.current, state.teams * state.rounds) / state.teams));
    const settingsTxt = leagueSettingsLabel();
    const hostBits = [state.teams + "-team"];
    if (settingsTxt) hostBits.push(settingsTxt);
    else hostBits.push(state.sf ? "SF" : "PPR");
    hostBits.push("round " + rd);
    document.getElementById("hostSub").textContent = hostBits.join(" · ");
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
      log.innerHTML = '<div class="empty-log">Waiting on pick 1.01. Simulate or hit Auto - the overlay never submits to the host.</div>';
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
    const me = all.filter(function (t) { return t.isMe; })[0] || all[0];
    document.getElementById("rosterChip").textContent = String(myPicks().length);
    const letter = me ? gradeLetter(me.grade.score) : "-";
    const chip = document.getElementById("gradesChip");
    chip.textContent = myPicks().length ? letter : "-";
    chip.style.color = myPicks().length && me ? gradeCol(me.grade.score) : "";
    const sumBtn = document.getElementById("sumBtn");
    if (sumBtn) sumBtn.hidden = !(draftDone() || myPicks().length);
    document.querySelectorAll(".tab-btn").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-tab") === state.tab);
    });
    const controls = document.getElementById("boardControls");
    if (controls) controls.hidden = state.tab !== "board";
    document.querySelectorAll("#posFilters [data-pos]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-pos") === state.pos ? "true" : "false");
    });
    const sortSel = document.getElementById("sortSel");
    if (sortSel && sortSel.value !== state.sort) sortSel.value = state.sort;
    const body = document.getElementById("ovBody");
    if (state.tab === "roster") body.innerHTML = renderRoster();
    else if (state.tab === "grades") body.innerHTML = renderGrades();
    else body.innerHTML = renderBoard();
    if (compareIds.length === 2) openCompare();
    else {
      const modal = document.getElementById("cmpModal");
      if (modal) modal.hidden = true;
    }
    const simBtn = document.getElementById("simBtn");
    if (simBtn) simBtn.disabled = draftDone() || state.live;
    if (EMBEDDED) {
      paintSyncChip();
      paintLiveClock();
    }
  }

  function paint() {
    availCache = null;
    cliffLeft = null;
    if (!EMBEDDED) {
      document.getElementById("stage").setAttribute("data-platform", state.platform);
      renderHost();
    }
    paintLeagueChrome();
    renderOverlay();
    paintSummary();
  }
  function render() {
    if (!EMBEDDED) {
      paint();
      return;
    }
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(function () {
      renderQueued = false;
      paint();
    });
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
  const posFilters = document.getElementById("posFilters");
  if (posFilters) posFilters.addEventListener("click", function (e) {
    const pos = e.target.closest("[data-pos]");
    if (!pos) return;
    state.pos = pos.getAttribute("data-pos");
    render();
  });
  const sortSelEl = document.getElementById("sortSel");
  if (sortSelEl) sortSelEl.addEventListener("change", function () {
    state.sort = sortSelEl.value || "rec";
    render();
  });
  const searchInp = document.getElementById("searchInp");
  if (searchInp) searchInp.addEventListener("input", function () {
    state.query = searchInp.value || "";
    render();
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
    const cmpModal = document.getElementById("cmpModal");
    if (cmpModal) {
      cmpModal.addEventListener("click", function (e) {
        if (e.target === cmpModal || e.target.closest("#drCmpClose") || e.target.closest("[data-cmp-close]")) {
          closeCompare();
        }
      });
    }
    const sumModal = document.getElementById("sumModal");
    if (sumModal) {
      sumModal.addEventListener("click", function (e) {
        if (e.target === sumModal || e.target.closest("#drSumClose") || e.target.closest("[data-sum-close]")) {
          closeSummary();
        }
      });
    }
  document.getElementById("ovBody").addEventListener("click", function (e) {
    if (e.target.closest("[data-open-summary]")) {
      e.preventDefault();
      openSummary();
      return;
    }
    const cmp = e.target.closest("[data-cmp]");
    if (cmp) {
      e.preventDefault();
      e.stopPropagation();
      toggleCompare(cmp.getAttribute("data-cmp"));
      return;
    }
    const row = e.target.closest(".ba-row");
    if (row) {
      if (EMBEDDED || state.live) return;
      draftToMe(row.getAttribute("data-id"));
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
    const leg = e.target.closest("[data-legslot]");
    if (leg) {
      const slot = +leg.getAttribute("data-legslot");
      state.expanded = state.expanded === slot ? null : slot;
      render();
    }
  });

  if (!EMBEDDED) clockTimer = setInterval(function () {
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
      const you = !!state.slotAuto && i === state.mySlot;
      const nm = state.teamNames && state.teamNames[i];
      o.textContent = (nm ? i + " · " + nm : String(i)) + (you ? " (you)" : "");
      if (i === state.mySlot) o.selected = true;
      sel.appendChild(o);
    }
    const lab = document.getElementById("slotLab");
    if (lab) lab.hidden = !EMBEDDED || !!state.slotAuto || !lastLiveDetail;
  }

  function liveClockSeconds() {
    if (state.hostClock == null || !isFinite(Number(state.hostClock))) return null;
    const elapsed = Math.floor((Date.now() - (state.hostClockAt || Date.now())) / 1000);
    return Math.max(0, Number(state.hostClock) - elapsed);
  }

  function applyHostClock(detail) {
    if (!detail || detail.clockSeconds == null || !isFinite(Number(detail.clockSeconds))) return;
    state.hostClock = Number(detail.clockSeconds);
    state.hostClockAt = Number(detail.clockAt || Date.now());
    if (detail.pickTimer != null && isFinite(Number(detail.pickTimer))) {
      state.pickTimer = Number(detail.pickTimer);
    }
    paintLiveClock();
  }

  function paintLiveClock() {
    const el = document.getElementById("ovOtc");
    if (!el) return;
    if (!EMBEDDED || !state.live) {
      el.hidden = true;
      return;
    }
    el.hidden = false;
    const done = draftDone();
    const slot = done ? null : ownerOf(state.current);
    const you = !done && slot === state.mySlot && state.slotAuto;
    el.classList.toggle("is-you", you);
    const pickEl = document.getElementById("ovOtcPick");
    const whoEl = document.getElementById("ovOtcWho");
    const clockEl = document.getElementById("ovOtcClock");
    if (pickEl) pickEl.textContent = done ? "Final" : pickLabel(state.current);
    if (whoEl) {
      whoEl.textContent = done
        ? "Draft complete"
        : (you ? "You are on the clock" : ("On the clock: " + teamName(slot)));
    }
    if (clockEl) {
      const secs = liveClockSeconds();
      if (done || secs == null) {
        clockEl.textContent = done ? "0:00" : "-:--";
      } else {
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        clockEl.textContent = m + ":" + String(s).padStart(2, "0");
      }
    }
  }

  function formatSyncChip(ok) {
    const plat = String(state.platform || "LIVE").replace(/[^a-z]/gi, "").toUpperCase() || "LIVE";
    const n = (state.picks || []).length;
    if (ok === false) return plat + " · …";
    const parts = [plat];
    if (n) parts.push(String(n));
    if (state.slotAuto && state.mySlot) parts.push("YOU " + state.mySlot);
    else if (!n) parts.push("LIVE");
    return parts.join(" · ");
  }

  function paintSyncChip(ok) {
    const chip = document.getElementById("syncChip");
    if (!chip) return;
    const synced = ok == null ? state.syncOk : !!ok;
    state.syncOk = synced;
    chip.innerHTML = "<i aria-hidden=\"true\"></i><span class=\"sync-chip-txt\">" + esc(formatSyncChip(synced)) + "</span>";
    chip.style.color = synced ? "" : "var(--warn)";
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
    if (detail.sf != null) state.sf = !!detail.sf;
    if (detail.scoringType && !state.leagueKind && !state.formatLabel) {
      state.draftType = (window.BRDraftSlot && BRDraftSlot.normDraftType)
        ? BRDraftSlot.normDraftType(detail.scoringType)
        : (String(detail.scoringType).toLowerCase() === "rookie" ? "rookie"
          : (String(detail.scoringType).toLowerCase() === "startup"
            || String(detail.scoringType).toLowerCase() === "dynasty") ? "startup" : "redraft");
    }
    fillAdpSel();
    players = rows.map(function (p) {
      const pos = normPos(p.pos || p.position || "RB");
      return {
        id: String(p.id),
        name: p.name,
        pos: pos,
        position: pos,
        team: p.team || "FA",
        adp: Number(p.adp) || 999,
        val: Number(p.val) || 0,
        ppg: p.ppg == null ? 0 : Number(p.ppg),
        age: p.age == null ? 0 : Number(p.age),
        bye: p.bye == null ? 0 : Number(p.bye),
        bye_week: p.bye_week != null ? Number(p.bye_week) : (p.bye == null ? 0 : Number(p.bye)),
        proj_ppg: p.proj_ppg != null ? Number(p.proj_ppg) : (p.ppg == null ? null : Number(p.ppg)),
        proj_pts: p.proj_pts == null ? null : Number(p.proj_pts),
        last_ppg: p.last_ppg == null ? null : Number(p.last_ppg),
        ppg_season: p.ppg_season || "",
        vorp: p.vorp == null ? null : Number(p.vorp),
        market: p.market == null ? null : Number(p.market),
        years_exp: p.years_exp == null ? null : Number(p.years_exp),
        is_rookie: !!p.is_rookie,
        injury: p.injury || "",
        headshot: p.headshot || "",
        tier: p.tier || 6,
        rank_change_7d: p.rank_change_7d == null ? null : Number(p.rank_change_7d),
        breakout_score: p.breakout_score == null ? null : Number(p.breakout_score),
        projected_role: p.projected_role || "",
      };
    });
    byId = {};
    byName = {};
    players.forEach(function (p) {
      if (p.id) byId[String(p.id)] = p;
      byName[normName(p.name)] = p;
      if (normPos(p.pos) === "DEF" && p.team && p.team !== "FA") {
        byId[String(p.team)] = byId[String(p.team)] || p;
        byName[normName(p.team + " dst")] = p;
        byName[normName(p.team + " def")] = p;
        byName[normName(p.team + " d/st")] = p;
      }
    });
    state.sitePool = true;
    let maxVal = 1;
    players.forEach(function (p) { if (p.val > maxVal) maxVal = p.val; });
    state.valCap = maxVal;
    lastLiveFp = "";
    if (lastLiveDetail) ingestLive(lastLiveDetail);
    else render();
  }

  function isCompletedHostPick(raw) {
    if (!raw) return false;
    const name = String(raw.playerName || raw.name || "").trim();
    if (/^pick\s*#?\s*\d+$/i.test(name)) return false;
    const pid = raw.playerId != null && String(raw.playerId) !== "" ? String(raw.playerId).trim() : "";
    const pidOk = pid && pid !== "0" && pid !== "-1" && pid !== "null" && pid !== "None";
    if (pidOk && byId[pid]) return true;
    if (name) return true;
    return false;
  }

  function matchAbbrevName(name, pos, nfl) {
    const m = String(name || "").trim().match(/^([A-Za-z])\.?\s+([A-Za-z][A-Za-z.'\-]+)$/);
    if (!m || !players || !players.length) return null;
    const initial = m[1].toLowerCase();
    const last = normName(m[2]);
    const wantPos = String(pos || "").toUpperCase();
    const wantNfl = String(nfl || "").toUpperCase().slice(0, 3);
    const hits = players.filter(function (p) {
      const parts = String(p.name || "").trim().split(/\s+/);
      if (parts.length < 2) return false;
      if (normName(parts[parts.length - 1]) !== last) return false;
      if (String(parts[0]).charAt(0).toLowerCase() !== initial) return false;
      if (wantPos && wantPos !== "WR" && p.pos && p.pos !== wantPos) return false;
      if (wantNfl && wantNfl.length === 3 && p.team && p.team !== "FA" && p.team !== wantNfl) return false;
      return true;
    });
    if (hits.length === 1) return hits[0];
    if (hits.length > 1 && wantNfl) {
      const nflHits = hits.filter(function (p) { return p.team === wantNfl; });
      if (nflHits.length === 1) return nflHits[0];
    }
    return hits[0] || null;
  }

  function matchLivePlayer(raw) {
    const pid = raw && raw.playerId != null && String(raw.playerId) !== "" ? String(raw.playerId) : "";
    if (pid && byId[pid]) return byId[pid];
    const name = (raw && (raw.playerName || raw.name)) || "";
    const key = normName(name);
    if (key && byName[key]) return byName[key];
    const pos = normPos((raw && (raw.pos || raw.position)) || "");
    const nfl = String((raw && (raw.nflTeam || raw.team || raw.proTeam)) || "").toUpperCase().slice(0, 3) || "FA";
    if (pos === "DEF" && nfl && nfl !== "FA") {
      const defHit = players.filter(function (p) { return normPos(p.pos) === "DEF" && p.team === nfl; })[0];
      if (defHit) return defHit;
    }
    const abbrev = matchAbbrevName(name, pos, nfl === "FA" ? "" : nfl);
    if (abbrev) return abbrev;
    const usePos = POS[pos] ? pos : "WR";
    const pn = Number((raw && (raw.overallPickNumber || raw.pick_no)) || 0) || state.current;
    const stub = {
      id: "live-" + (pid || key || pn),
      name: name || ("Pick " + pn),
      pos: usePos,
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

  function applyLeagueSettings(detail) {
    if (!detail) return;
    if (detail.sf != null) state.sf = !!detail.sf;
    if (detail.leagueName) state.leagueName = String(detail.leagueName);
    if (detail.leagueId) state.leagueId = String(detail.leagueId);
    if (detail.season) state.season = Number(detail.season) || state.season;
    if (detail.draftType) {
      state.draftType = (window.BRDraftSlot && BRDraftSlot.normDraftType)
        ? BRDraftSlot.normDraftType(detail.draftType)
        : (String(detail.draftType).toLowerCase() === "rookie" ? "rookie"
          : (String(detail.draftType).toLowerCase() === "startup"
            || String(detail.draftType).toLowerCase() === "dynasty") ? "startup" : "redraft");
    }
    if (detail.leagueKind) state.leagueKind = String(detail.leagueKind);
    if (detail.formatLabel) state.formatLabel = String(detail.formatLabel);
    if (detail.orderFormat) state.orderFormat = String(detail.orderFormat);
    if (detail.orderLabel) state.orderLabel = String(detail.orderLabel);
    if (detail.bestBall != null) state.bestBall = !!detail.bestBall;
    if (detail.slotToRosterId && typeof detail.slotToRosterId === "object") {
      state.slotToRosterId = detail.slotToRosterId;
    }
    if (detail.ppr != null && isFinite(Number(detail.ppr))) state.ppr = Number(detail.ppr);
    if (detail.tep != null && isFinite(Number(detail.tep))) state.tep = Number(detail.tep);
    if (detail.passTd != null && isFinite(Number(detail.passTd))) state.passTd = Number(detail.passTd);
    if (detail.roster && typeof detail.roster === "object") {
      const rs = detail.roster;
      const starters = (rs.QB || 0) + (rs.RB || 0) + (rs.WR || 0) + (rs.TE || 0)
        + (rs.FLEX || 0) + (rs.SF || 0) + (rs.RB_WR || 0) + (rs.WR_TE || 0) + (rs.RB_TE || 0);
      if (starters >= 4) {
        state.roster = {
          QB: Number(rs.QB) || 0,
          SF: Number(rs.SF) || 0,
          RB: Number(rs.RB) || 0,
          WR: Number(rs.WR) || 0,
          TE: Number(rs.TE) || 0,
          FLEX: Number(rs.FLEX) || 0,
          RB_WR: Number(rs.RB_WR) || 0,
          WR_TE: Number(rs.WR_TE) || 0,
          RB_TE: Number(rs.RB_TE) || 0,
          K: Number(rs.K) || 0,
          DEF: Number(rs.DEF) || 0,
          BN: Number(rs.BN) || 0,
        };
        if ((state.roster.SF || 0) > 0) state.sf = true;
      }
    }
  }

  function leagueSettingsLabel() {
    const format = state.formatLabel
      || (window.BRDraftSlot && BRDraftSlot.formatKindLabel
        ? BRDraftSlot.formatKindLabel(state.leagueKind, state.draftType)
        : (state.draftType === "rookie" ? "Rookie"
          : (state.draftType === "startup" || state.draftType === "dynasty" ? "Dynasty" : "Redraft")));
    if (window.BRDraftSlot && BRDraftSlot.settingsLabel && state.roster) {
      return BRDraftSlot.settingsLabel(state.roster, {
        ppr: state.ppr, tep: state.tep, passTd: state.passTd, format: format,
        order: state.orderLabel || "",
        bestBall: state.bestBall,
      });
    }
    const bits = [format];
    if (state.orderLabel) bits.push(state.orderLabel);
    if (state.bestBall) bits.push("Best Ball");
    if (state.sf) bits.push("SF");
    return bits.filter(Boolean).join(" · ");
  }

  function paintLeagueChrome() {
    const nameEl = document.getElementById("ovLeagueName");
    const settingsEl = document.getElementById("ovLeagueSettings");
    if (nameEl && state.leagueName) nameEl.textContent = state.leagueName;
    if (!settingsEl) return;
    const bits = [];
    if (state.teams >= 2) bits.push(state.teams + "tm");
    const settingsTxt = leagueSettingsLabel();
    if (settingsTxt) bits.push(settingsTxt);
    else if (state.sf) bits.push("SF");
    const label = bits.join(" · ");
    settingsEl.textContent = label;
    settingsEl.hidden = !label;
  }

  function toggleCompare(id) {
    id = String(id || "");
    if (!id) return;
    const idx = compareIds.indexOf(id);
    if (idx >= 0) {
      compareIds.splice(idx, 1);
    } else if (compareIds.length >= 2) {
      compareIds = [id];
    } else {
      compareIds.push(id);
    }
    render();
  }

  function closeCompare() {
    compareIds = [];
    const modal = document.getElementById("cmpModal");
    if (modal) modal.hidden = true;
    render();
  }

  function infoIcon(tip) {
    return '<span class="dr-info" tabindex="0" role="button" aria-label="' + esc(tip) + '" data-tip="' + esc(tip) + '">i</span>';
  }

  function expLabel(p) {
    if (!p) return "";
    if (p.is_rookie) return "Rookie";
    const ye = Number(p.years_exp);
    if (!isFinite(ye) || ye < 0) return "";
    if (ye === 0) return "Rookie";
    return ye + " yr";
  }

  function posRankOf(p) {
    const pos = normPos(p && (p.pos || p.position));
    if (!pos) return { label: "", n: null };
    const ranked = players.filter(function (x) {
      return normPos(x.pos || x.position) === pos;
    }).slice().sort(function (a, b) {
      if (pos === "K" || pos === "DEF") {
        return (window.BROverlayScore && BROverlayScore.sortKdef)
          ? BROverlayScore.sortKdef(a, b)
          : ((a.adp || 999) - (b.adp || 999)) || ((b.ppg || 0) - (a.ppg || 0));
      }
      return (Number(b.val) || 0) - (Number(a.val) || 0);
    });
    const i = ranked.findIndex(function (x) { return String(x.id) === String(p.id); });
    if (i < 0) return { label: "", n: null };
    return { label: pos + (i + 1), n: i + 1 };
  }

  function fmtSigned(n, digits) {
    if (n == null || !isFinite(Number(n))) return "-";
    const x = Number(n);
    const s = digits != null ? x.toFixed(digits) : String(Math.round(x));
    if (Number(s) === 0) return digits != null ? Number(0).toFixed(digits) : "0";
    return (Number(s) > 0 ? "+" : "") + s;
  }

  function draftPlayerFacts(p, pool) {
    const hit = (pool || []).filter(function (x) { return String(x.id) === String(p.id); })[0] || p;
    const adp = Number(p.adp);
    const adpN = isFinite(adp) && adp < 900 ? adp : null;
    const scoring = { ppr: state.ppr, tep: state.tep, passTd: state.passTd };
    const C = window.DraftBoardCore;
    let projPpg = C && C.scoringProjPpg ? C.scoringProjPpg(p, scoring) : null;
    if (projPpg == null) projPpg = p.proj_ppg != null ? Number(p.proj_ppg) : (p.ppg != null ? Number(p.ppg) : null);
    let projPts = C && C.scoringProjPts ? C.scoringProjPts(p, scoring) : null;
    if (projPts == null) projPts = p.proj_pts != null ? Number(p.proj_pts) : null;
    const lastPpg = p.last_ppg != null && isFinite(Number(p.last_ppg)) ? Number(p.last_ppg) : null;
    const pr = posRankOf(p);
    let survive = null;
    const ctx = scoreCtx();
    if (window.BROverlayScore && BROverlayScore.recWaitPickNo && BROverlayScore.availProb) {
      const next = BROverlayScore.recWaitPickNo(ctx);
      if (next) {
        const pct = BROverlayScore.availProb(p, next, ctx, players);
        if (pct != null && isFinite(Number(pct))) survive = Math.round(Number(pct));
      }
    }
    return {
      rec: hit._rank != null ? hit._rank : null,
      ps: hit._psShow != null ? hit._psShow : (hit._ps != null ? hit._ps : null),
      value: Number(p.val) || 0,
      projPpg: projPpg != null && isFinite(Number(projPpg)) ? Number(projPpg) : null,
      lastPpg: lastPpg,
      ppgSeason: p.ppg_season || "",
      vor: hit._vor != null && isFinite(Number(hit._vor)) ? Number(hit._vor) : null,
      vorp: p.vorp != null && isFinite(Number(p.vorp)) ? Number(p.vorp) : null,
      adp: adpN,
      vsAdp: adpN != null ? (state.current - adpN) : null,
      posRank: pr.label,
      posRankN: pr.n,
      bye: p.bye_week != null || p.bye != null ? Number(p.bye_week != null ? p.bye_week : p.bye) : null,
      age: p.age != null ? Number(p.age) : null,
      survive: survive,
      projPts: projPts != null && isFinite(Number(projPts)) ? Number(projPts) : null,
      market: p.market != null && isFinite(Number(p.market)) ? Number(p.market) : null,
      exp: expLabel(p),
      injury: p.injury || "",
    };
  }

  function openCompare() {
    const p1 = byId[String(compareIds[0])];
    const p2 = byId[String(compareIds[1])];
    const modal = document.getElementById("cmpModal");
    const card = document.getElementById("cmpCard");
    if (!p1 || !p2 || !modal || !card) return;
    const counts = posCounts(myPicks());
    const pool = rankedPool(counts, state.current);
    function cmpCol(p, other) {
      const f = draftPlayerFacts(p, pool);
      const o = draftPlayerFacts(other, pool);
      const ps = f.ps;
      function statRow(lbl, val, oval, higherBetter, fmtFn, tip) {
        if (val == null && oval == null) return "";
        const vStr = fmtFn ? fmtFn(val) : (val != null ? String(val) : "-");
        const win = val != null && oval != null && (higherBetter ? val > oval : val < oval);
        return '<div class="dr-cmp-stat' + (win ? " win" : "") + '">'
          + '<span class="dr-cmp-stat-lbl"' + (tip ? ' title="' + esc(tip) + '"' : "") + ">" + esc(lbl) + "</span>"
          + '<span class="dr-cmp-stat-val">' + esc(vStr) + "</span></div>";
      }
      const sc = ps != null ? psColor(ps) : "var(--text-muted)";
      const photo = hsUrl(p)
        ? '<img class="dr-cmp-hs" src="' + esc(hsUrl(p)) + '" alt="">'
        : hsMark(p, "hs-sm");
      const metaBits = [p.team || "", f.exp, (f.age ? "Age " + f.age.toFixed(0) : ""), f.injury].filter(Boolean);
      return '<div class="dr-cmp-player">'
        + '<div class="dr-cmp-top">' + photo
        + '<div><div class="dr-cmp-name"><span class="dr-posbadge" style="background:' + (POS[p.pos] || POS.BN) + '">' + esc(p.pos) + "</span> " + esc(p.name) + "</div>"
        + '<div class="dr-cmp-meta">' + esc(metaBits.join(" · ")) + "</div>"
        + "</div></div>"
        + '<div class="dr-cmp-ps" style="color:' + sc + '">' + (ps != null ? Math.round(ps) : "-") + "</div>"
        + '<div class="dr-cmp-ps-lbl">Pick Score'
        + infoIcon("How good is this player at this pick? Absolute 0-100 quality (value, VOR, ADP, tier, need, projected points), shown relative to the best player still available. Not Recommendation Rank, and not a count of which compare rows you win.")
        + "</div>"
        + '<div class="dr-cmp-stats">'
        + statRow("Value", f.value, o.value, true, function (x) { return x != null ? String(Math.round(x)) : "-"; })
        + statRow("Proj PPG", f.projPpg, o.projPpg, true, function (x) { return x != null ? x.toFixed(1) : "N/A"; })
        + (f.lastPpg != null || o.lastPpg != null ? statRow((f.ppgSeason || "Last") + " PPG", f.lastPpg, o.lastPpg, true, function (x) { return x != null ? x.toFixed(1) : "-"; }) : "")
        + statRow("VOR", f.vor, o.vor, true, function (x) { return x != null ? fmtSigned(x, Number.isInteger(x) ? 0 : 1) : "-"; })
        + (f.vorp != null || o.vorp != null ? statRow("VORP", f.vorp, o.vorp, true, function (x) { return x != null ? fmtSigned(x, Number.isInteger(x) ? 0 : 1) : "N/A"; }) : "")
        + statRow("ADP", f.adp, o.adp, false, function (x) { return x != null ? Number(x).toFixed(1) : "N/A"; })
        + statRow("vs ADP", f.vsAdp, o.vsAdp, true, function (x) { return fmtSigned(Math.round(x), 0); })
        + (f.posRank || o.posRank ? statRow("Pos Rank", f.posRankN, o.posRankN, false, function (x) {
          if (x == null) return "-";
          if (f.posRankN === x && f.posRank) return f.posRank;
          if (o.posRankN === x && o.posRank) return o.posRank;
          return String(x);
        }) : "")
        + (f.bye != null || o.bye != null ? statRow("Bye", f.bye, o.bye, false, function (x) { return x != null ? String(x) : "-"; }) : "")
        + (f.rec != null || o.rec != null ? statRow("REC", f.rec, o.rec, false, function (x) { return x != null ? "#" + x : "-"; }, "Recommendation Rank - who to draft now (roster-aware order, not a grade)") : "")
        + (f.survive != null || o.survive != null ? statRow("Survive", f.survive, o.survive, true, function (x) { return x != null ? x + "%" : "-"; }) : "")
        + (f.projPts != null || o.projPts != null ? statRow("Proj Pts", f.projPts, o.projPts, true, function (x) { return x != null ? String(Math.round(x)) : "-"; }) : "")
        + (f.market != null || o.market != null ? statRow("Mkt vs ADP", f.market, o.market, true, function (x) { return fmtSigned(Math.round(x), 0); }) : "")
        + "</div></div>";
    }
    card.innerHTML = '<button type="button" class="dr-cmp-close" id="drCmpClose" data-cmp-close="1" aria-label="Close">&times;</button>'
      + '<div class="dr-cmp-title" id="cmpTitle">Compare Players</div>'
      + '<div class="dr-cmp-cols">' + cmpCol(p1, p2) + cmpCol(p2, p1) + "</div>";
    modal.hidden = false;
  }

  function ingestLive(detail) {
    if (!detail) return;
    const fp = liveFingerprint(detail);
    if (fp === lastLiveFp && lastLiveDetail) {
      if (detail.syncText) setSyncStatus(true, detail.syncText);
      return;
    }
    lastLiveFp = fp;
    lastLiveDetail = detail;
    state.live = true;
    stopAuto();
    if (detail.platform) state.platform = String(detail.platform).toLowerCase();
    applyLeagueSettings(detail);
    if (detail.inProgress != null) state.hostInProgress = !!detail.inProgress;
    if (detail.drafted != null) state.hostDrafted = !!detail.drafted;
    if (detail.teams) state.teams = Math.max(2, Number(detail.teams) || state.teams);
    if (detail.rounds) {
      const r = Math.max(1, Number(detail.rounds) || 0);
      const maxPn = Array.isArray(detail.picks)
        ? detail.picks.reduce(function (m, p) {
            const n = Number(p && (p.overallPickNumber || p.pick_no)) || 0;
            return n > m ? n : m;
          }, 0)
        : 0;
      const inferred = state.teams ? Math.ceil(maxPn / state.teams) : 0;
      if (r >= 6 && !(r === inferred && r < 10)) state.rounds = Math.max(state.rounds, r);
    }
    if (detail.mySlot) {
      state.mySlot = Math.max(1, Math.min(state.teams, Number(detail.mySlot)));
      state.slotAuto = true;
    }
    if (detail.teamNames && typeof detail.teamNames === "object") {
      state.teamNames = Object.assign({}, state.teamNames, detail.teamNames);
    }
    if (detail.pickOwners && typeof detail.pickOwners === "object") {
      state.pickOwners = detail.pickOwners;
    }
    applyHostClock(detail);
    const raw = Array.isArray(detail.picks) ? detail.picks.slice() : [];
    raw.sort(function (a, b) {
      return (Number(a.overallPickNumber || a.pick_no || 0) - Number(b.overallPickNumber || b.pick_no || 0));
    });
    state.picks = [];
    state.drafted = {};
    raw.forEach(function (rp) {
      const pn = Number(rp.overallPickNumber || rp.pick_no || 0);
      if (!pn || !isCompletedHostPick(rp)) return;
      const p = matchLivePlayer(rp);
      if (!p || /^pick\s*#?\s*\d+$/i.test(String(p.name || ""))) return;
      const explicit = Number(rp.slot || rp.draftSlot || rp.draft_slot || 0);
      const mapped = Number(state.pickOwners && state.pickOwners[pn]);
      const slot = mapped >= 1 && mapped <= state.teams
        ? mapped
        : (explicit >= 1 && explicit <= state.teams ? explicit : ownerOf(pn));
      const counts = posCounts(teamPicks(slot));
      const need = needOf(counts, p.pos) > 0;
      const grade = pickLetter(pn - p.adp, need);
      const ps = pickScore(p, counts, pn);
      state.picks.push({ pn: pn, slot: slot, p: p, grade: grade, ps: ps });
      state.drafted[p.id] = true;
    });
    const lastMade = state.picks.length ? state.picks[state.picks.length - 1].pn : 0;
    const clockPn = Number(detail.current || detail.clockOverall || 0);
    if (clockPn >= 1) state.current = clockPn;
    else state.current = lastMade + 1;
    if (state.hostDrafted && lastMade && state.teams) {
      const inferred = Math.ceil(lastMade / state.teams);
      if (inferred >= 6 && inferred <= 30) state.rounds = inferred;
    }
    if (detail.clockSeconds == null) state.clock = CLOCK_START;
    fillSlotSel();
    paintSyncChip(true);
    maybeShowSummary();
    render();
  }

  function setSyncStatus(ok, text) {
    void text;
    paintSyncChip(!!ok);
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
  document.querySelector(".overlay").addEventListener("click", function (e) {
    if (e.target.closest("[data-open-summary]")) {
      e.preventDefault();
      openSummary();
      return;
    }
    const link = e.target.closest("[data-link]");
    if (!link || link.getAttribute("data-link") !== "sheet") return;
    if (EMBEDDED) {
      postToHost("open", { dest: "sheet" });
      return;
    }
    try {
      window.open(
        "https://www.brfantasyfootball.com/draft/cheat-sheet",
        "_blank",
        "noopener"
      );
    } catch (_e) { /* ignore */ }
  });
  const slotSel = document.getElementById("slotSel");
  if (slotSel) slotSel.addEventListener("change", function () {
    state.mySlot = Number(slotSel.value) || state.mySlot;
    state.slotAuto = false;
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
    if (msg.type === "clock") applyHostClock(msg);
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

  if (EMBEDDED) {
    setInterval(function () {
      if (!state.live || draftDone()) return;
      paintLiveClock();
    }, 1000);
  }
  if (!EMBEDDED) {
    buildPool();
    indexNames();
  }
  if (!EMBEDDED) {
    const savedSlot = Number(localStorage.getItem("br-da-slot") || 0);
    if (savedSlot) state.mySlot = savedSlot;
  }
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
