// Shared analytics terminology — keep in sync with utils/analytics_terminology.py
(function () {
  var C = { VALUE: 'VALUE', MARKET: 'MARKET', PROJECTION: 'PROJECTION', HISTORY: 'HISTORY' };
  var LABELS = {
    br_value: { label: 'BR Value', category: C.VALUE, tooltip: 'Dynasty trade value from the BR model for this league format.' },
    vor: { label: 'VOR', category: C.VALUE, tooltip: 'Value over replacement — points above a rosterable baseline at this position.' },
    trade_value: { label: 'Trade value', category: C.VALUE, tooltip: 'Fair-trade value used in the calculator and trade database.' },
    roster_value: { label: 'Roster value', category: C.VALUE, tooltip: 'Combined player and pick value on a team\'s roster.' },
    adp: { label: 'ADP', category: C.MARKET, tooltip: 'Average draft position from the selected consensus source.' },
    market_vs_adp: { label: 'Market vs ADP', category: C.MARKET, tooltip: 'Where independent market signals imply this player should be drafted relative to current ADP.' },
    market_movement: { label: 'Market movement', category: C.MARKET, tooltip: 'Recent change in trade frequency or market price for this player.' },
    trade_frequency: { label: 'Trade frequency', category: C.MARKET, tooltip: 'How often this player appears in real dynasty trades over a recent window.' },
    projected_ppg: { label: 'Proj PPG', category: C.PROJECTION, tooltip: 'Projected fantasy points per game for the current scoring settings.' },
    start_score: { label: 'Start score', category: C.PROJECTION, tooltip: 'Start/Sit recommendation strength for this week\'s lineup decision.' },
    playoff_odds: { label: 'Playoff odds', category: C.PROJECTION, tooltip: 'Simulated chance to make the playoffs from current rosters and schedule.' },
    historical_hit_rate: { label: 'Hist', category: C.HISTORY, tooltip: 'Historical top-12 chance for players with this career profile and situation. Not a ranking input.' },
    historical_comps: { label: 'Players like this', category: C.HISTORY, tooltip: 'Historical outcomes for players with a similar career arc and usage profile.' },
    adp_range_comps: { label: 'ADP range history', category: C.HISTORY, tooltip: 'Historical outcomes for players drafted in this ADP range, regardless of profile.' }
  };

  window.brAnalyticsLabel = function (key) {
    return (LABELS[key] && LABELS[key].label) || key;
  };

  window.brAnalyticsTooltip = function (key) {
    var e = LABELS[key];
    if (!e) return '';
    return e.category ? (e.category + ': ' + e.tooltip) : e.tooltip;
  };
})();
