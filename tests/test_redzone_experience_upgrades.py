"""Focused contracts for grouped Redzone experience upgrades."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "redzone.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()


def _node(source: str):
    out = subprocess.run(["node", "-e", source], check=True, text=True, capture_output=True)
    return json.loads(out.stdout)


def test_component_scorer_uses_one_total_source_and_preserves_precision():
    block = JS[JS.index("function _n("):JS.index("function _scoringForPid(")]
    cases = _node(f"""
      var _state={{scoring:{{}}}}; {block}
      const score={{rec:1,rec_yd:.1,pass_yd:.05,pass_td:4,pass_int:-2,
        bonus_rec_te:.5,rec_td:6,fgm_50p:5,sack:1,def_int:2,fum_rec:2,def_td:6,safe:2}};
      const lines=[
        [{{rec:1,rec_yds:12}},score,'WR'], [{{rec:1,rec_yds:5}},{{rec:.5,rec_yd:.1}},'WR'],
        [{{rec:1}},score,'TE'], [{{pass_yds:11}},score,'QB'], [{{rec:1,rec_yds:24,rec_td:1}},score,'WR'],
        [{{pass_td:1}},score,'QB'], [{{int:1}},score,'QB'], [{{fgm:1,fg_yds:52}},score,'K'],
        [{{sacks:1}},score,'DEF'], [{{def_int:1}},score,'DEF'], [{{rush_yds:-5}},{{rush_yd:.01}},'RB']
      ];
      console.log(JSON.stringify(lines.map(x=>_scoreContribution(...x))));
    """)
    assert [x["total"] for x in cases] == [2.2, 1, 1.5, .55, 9.4, 4, -2, 5, 1, 2, -.05]
    for result in cases:
        assert abs(sum(c["points"] for c in result["components"]) - result["total"]) < 1e-9
    assert "_lineToPts(L, s, pos) { return _scoreContribution" in block


def test_big_play_classifier_central_thresholds_and_independent_contributions():
    block = JS[JS.index("var _BIG_PLAY_THRESHOLDS"):JS.index("// \"MM:SS\"")]
    result = _node(f"""
      function _n(x){{return parseFloat(x||0)||0}}; {block}
      const ev=l=>({{contributions:[{{line:l,pts:0,breakdown:{{components:[]}}}}]}});
      const inputs=[{{rec_td:1}},{{rec:1,rec_yds:25}},{{carries:1,rush_yds:25}},{{pass_yds:45}},
        {{fgm:1,fg_yds:52}},{{int:1}},{{fum_lost:1}},{{def_td:1}},{{rec:1,rec_yds:7}},
        {{carries:1,rush_yds:4}},{{rec:1,rec_yds:7,red_zone:1}}];
      console.log(JSON.stringify(inputs.map(x=>_classifyBigPlay(ev(x)))));
    """)
    assert [x["isBig"] for x in result] == [True] * 8 + [False] * 3
    assert result[1]["category"] == "explosive_reception"
    assert result[2]["category"] == "explosive_rush"
    assert "(ev.pts || 0) >= 4" not in JS


def test_grouped_secondary_contributors_and_accessible_breakdown_contracts():
    assert "secondaryContributions: validContribs.filter" in JS
    assert "c.pid !== primary.pid && Math.abs(c.pts)" in JS
    assert "slice(0, 2)" in JS  # visibility is capped, canonical contributions are not
    assert "data-pid=\"' + c.pid" in JS
    assert "data-scoring-play" in JS and "aria-expanded" in JS
    assert "e.stopPropagation()" in JS and "e.key === 'Escape'" in JS
    assert ".rz-secondary" in CSS and ".rz-scoring-popover" in CSS


def test_starter_bench_and_richer_live_filter_labels_contracts():
    assert "tags.starters.has(String(rid) + ':' + pid)" in JS
    assert "return side + ' ' + (ev.starter ? 'STARTER' : 'BENCH')" in JS
    assert "if (!ev.rosterId || (!ev.mine && !ev.opp)) return ''" in JS
    assert "g.awayScore" in JS and "g.status.label" in JS and "'FINAL'" in JS
    assert "_fmtFantasyPrecise(pair[0].points)" in JS
    assert "m.league_name" in JS


def test_corrections_replace_canonical_history_and_dom_without_reordering():
    assert "_pbpHistory[hi] = contrib" in JS
    assert "_feed[i] = upd" in JS
    assert "old.replaceWith(fresh)" in JS
    assert "correctedUntil" in JS and "STAT CORRECTION" in JS
    assert "OVERTURNED" in JS and "NO PLAY" in JS
    assert "_feed = _chronoSort(_feed)" in JS
    assert "prefers-reduced-motion: reduce" in CSS
