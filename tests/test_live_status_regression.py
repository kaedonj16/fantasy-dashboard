from utils.utils import build_games_by_team, build_status_by_pid, STATUS_IN_PROGRESS, STATUS_NOT_STARTED


def _status(game):
    games = build_games_by_team([{"home": "BUF", "away": "MIA", "gameStatusCode": "1", **game}])
    return build_status_by_pid({"p": {"team": "BUF"}}, games, {}, 1)["p"]


def test_preseason_game_never_marks_player_live():
    assert _status({"seasonType": "pre"}) == STATUS_NOT_STARTED


def test_missing_or_unknown_season_type_is_conservative():
    assert _status({}) == STATUS_NOT_STARTED
    assert _status({"season_type": "exhibition"}) == STATUS_NOT_STARTED


def test_regular_and_postseason_games_can_be_live():
    assert _status({"seasonType": "regular"}) == STATUS_IN_PROGRESS
    assert _status({"game_type": "POST"}) == STATUS_IN_PROGRESS
