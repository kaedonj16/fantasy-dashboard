// Single source of truth for the draft-grade field curve.
//
// Loaded as a plain <script> in the draft room (exposes window.BRDraftGrade)
// and require()'d by the parity test in Node. The Python mirror in
// utils/draft_grade.py (dr_apply_field_curve) is pinned to THIS implementation
// by tests/test_draft_grade_curve_parity.py, so the browser (Draft Room) and
// server (Teams page /api/draft-grades) can never grade the same team
// differently again.
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BRDraftGrade = api;
})(typeof self !== 'undefined' ? self : this, function () {
  // ANCHOR 74 -> 70 from the letter-calibration backtest: at 74 the top THIRD of
  // every league landed in A-range (~31% of teams) - too generous. Anchoring the
  // average at a low B reserves A-range for ~the best 1-2 teams per league.
  var ANCHOR = 70;   // the field-average team lands at a low B
  // PTS 11 -> 9: grade separation only weakly predicts real success, so keep the
  // spread modest - an A needs ~+1.5 SD, not +1.
  var PTS = 9;       // grade points per standard deviation of real separation

  // Curve raw composite scores against the field so real separation reads on a
  // B-anchored scale, without letting best-of-field manufacture elite letters.
  //   rawScores  : number[] raw 0-100 composites, one per team
  //   roundsDone : completed draft rounds (drives early-draft damping)
  // Returns curved scores aligned to rawScores. Fewer than 3 teams -> unchanged
  // (no field to curve against). Uses round-half-up to match Python's
  // floor(x + 0.5) exactly.
  function curveFieldScores(rawScores, roundsDone) {
    var n = rawScores.length;
    if (n < 3) return rawScores.slice();

    var mean = 0;
    for (var i = 0; i < n; i++) mean += rawScores[i];
    mean /= n;
    var variance = 0;
    for (var j = 0; j < n; j++) { var d = rawScores[j] - mean; variance += d * d; }
    variance /= n;
    // Floor the spread so a near-identical field isn't blown apart while still
    // giving a readable range when teams genuinely differ.
    var effStd = Math.max(Math.sqrt(variance), 8);

    // Early-draft damping: a few picks per team is mostly noise, so the curve
    // spreads at half strength and ramps to full by round 6.
    var ramp = Math.max(0, Math.min(1, (roundsDone || 0) / 6));
    var ptsEff = PTS * (0.5 + 0.5 * ramp);

    return rawScores.map(function (raw) {
      var z = (raw - mean) / effStd;
      var curved = ANCHOR + z * ptsEff;
      // Best-of-field can't mint elite letters: the curve tops out just above
      // the raw composite, and the A band requires real raw quality - the best
      // draft in a mediocre room reads B+/A-, not A+.
      curved = Math.min(curved, raw + 8);
      if (curved >= 85 && raw < 80) curved = 84;
      curved = Math.max(0, Math.min(100, curved));
      return Math.floor(curved + 0.5);
    });
  }

  return { curveFieldScores: curveFieldScores };
});
