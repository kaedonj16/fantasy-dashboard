"""Regression coverage for normalized NFL and fantasy matchup navigation state."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "redzone.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()


def node_json(source):
    return json.loads(subprocess.run(["node", "-e", source], check=True, capture_output=True, text=True).stdout)


def test_normalized_game_status_authority_and_provider_code_semantics():
    resolver = JS[JS.index("function _resolveGameStatus("):JS.index("function _normalizeGames(")]
    normalizer = JS[JS.index("function _normalizeGames("):JS.index("function _gameStatus(")]
    out = node_json(f"""
      function _fmtQuarter(q){{return q ? (/^Q/.test(String(q)) ? String(q) : 'Q'+q) : ''}}
      function _fmtKickoff(){{return 'Sun 4:25 PM'}}
      {resolver}
      const rows=[
        [{{game_status:'In Progress',game_code:'1',game_quarter:'4',game_clock:'2:18',down:'2',distance:'6',yard_line:'SF 15',possession:'SF'}},true],
        [{{game_status:'Halftime',game_code:'1',down:'1',distance:'10'}},true],
        [{{game_status:'Final',game_code:'2',down:'2',distance:'6',yard_line:'SF 15',possession:'SF'}},true],
        [{{game_status:'In Progress',game_code:'2',game_quarter:'3',game_clock:'7:42'}},true],
        [{{game_status:'Final',game_code:'2'}},false],
        [{{game_status:'Scheduled',game_code:'2'}},true]
      ];
      console.log(JSON.stringify(rows.map(x=>_resolveGameStatus(x[0],x[1]))));
    """)
    assert [x["type"] for x in out] == ["live", "halftime", "final", "live", "unknown", "pregame"]
    assert out[0]["down"] == "2" and out[0]["yardLine"] == "SF 15"
    assert not out[2]["down"] and not out[2]["yardLine"] and not out[2]["possession"]
    assert "code === '2'" not in resolver

    games = node_json(f"""
      function _fmtQuarter(q){{return q ? 'Q'+q : ''}}; function _fmtKickoff(){{return 'Sun 4:25 PM'}}
      {resolver} {normalizer}
      const data={{games:{{g1:{{game_status:'In Progress',game_code:'1',game_quarter:'3',game_clock:'7:42'}},g2:{{game_status:'Final',game_code:'2'}}}},player_info:{{stale:{{game_id:'g1',game_status:'Final',game_code:'2'}}}}}};
      console.log(JSON.stringify(_normalizeGames(data)));
    """)
    assert games["g1"]["status"]["type"] == "live"
    assert games["g2"]["status"]["type"] == "final"


def test_fantasy_matchup_state_counts_upcoming_live_final_bye_and_unknown():
    block = JS[JS.index("function _rosterGameCounts("):JS.index("function _playersLeft(")]
    states = node_json(f"""
      let status={{}}; function _gameStatus(pid){{return {{type:status[pid]||'unknown'}}}}; {block}
      const side=(prefix, kinds)=>({{starters:kinds.map((k,i)=>{{status[prefix+i]=k;return prefix+i}})}});
      const result=[];
      result.push(_fantasyMatchupState(side('a',['pregame','pregame','pregame','pregame','pregame']),side('b',['pregame','pregame','pregame','pregame','pregame','pregame'])));
      result.push(_fantasyMatchupState(side('c',['live','live','pregame']),side('d',['live','pregame'])));
      result.push(_fantasyMatchupState(side('e',['final']),side('f',['pregame'])));
      result.push(_fantasyMatchupState(side('g',['final','bye']),side('h',['final'])));
      result.push(_fantasyMatchupState(side('i',['final']),side('j',['pregame','pregame','pregame','pregame'])));
      result.push(_fantasyMatchupState(side('k',['final']),side('l',['unknown'])));
      console.log(JSON.stringify(result));
    """)
    assert states[0]["type"] == "to-play" and (states[0]["left"]["toPlay"], states[0]["right"]["toPlay"]) == (5, 6)
    assert states[1]["type"] == "live" and (states[1]["left"]["playingNow"], states[1]["right"]["playingNow"]) == (2, 1)
    assert states[2]["type"] == "to-play"
    assert states[3]["type"] == "final" and states[3]["left"]["toPlay"] == 0
    assert states[4]["type"] == "to-play" and states[4]["right"]["toPlay"] == 4
    assert states[5]["type"] == "unknown"


def test_game_strip_filter_sync_accessibility_and_mobile_layout_contracts():
    assert "function _renderGameStrip()" in JS
    assert "data-game-id=\"all\"" in JS and "aria-pressed" in JS
    assert "_filters.nfl = btn.dataset.gameId" in JS
    assert "scrollIntoView({ inline: 'nearest', block: 'nearest' })" in JS
    assert "fpRow('Matchup', 'nfl', nOpts)" in JS
    assert JS.index("+ _renderGameStrip()") < JS.index("+ panel;")
    assert "overflow-x: auto" in CSS and "scroll-snap-type: x proximity" in CSS
    assert "flex-wrap: nowrap" in CSS and "-webkit-overflow-scrolling: touch" in CSS


def test_selected_scoreboard_and_pills_share_game_object_and_final_hides_situation():
    assert "return _gamesById[gid]" in JS
    assert "var games = _nflMatchupOptions()" in JS
    assert "g.status.type !== 'live'" in JS
    assert "type === 'live' ? (row.down || '') : ''" in JS
    assert ".rz-nfl-board.is-final" in CSS
