"""ScoreZone TD clip matcher: gate, storage, payload attach, real shapes.

Fixture captions are the REAL @NFL captions from the 2026-10-25 Rams at
Raiders backtest (plus the 2026-10-29 TNF Golden TD), with the real tweet
ids/timestamps wherever the backtest recovered them.
"""
from __future__ import annotations

import pytest

import utils.scorezone_clips as clips
import utils.scorezone_store as store

RAMS_GAME = "20261025_LAR@LV"
TNF_GAME = "20261029_CAR@GB"

PID_TO_NAME = {
    "stafford": "Matthew Stafford",
    "allen-d": "Davis Allen",
    "nacua": "Puka Nacua",
    "kyren": "Kyren Williams",
    "cousins": "Kirk Cousins",
    "tucker": "Tre Tucker",
    "jeanty": "Ashton Jeanty",
    "corum": "Blake Corum",
    "love": "Jordan Love",
    "golden": "Matthew Golden",
}


def _row(play_id, pid, stat, wallclock, quarter="", clock=""):
    return {
        "play_id": play_id, "pid": pid, "is_td": True,
        "stat_line": stat, "wallclock": wallclock,
        "quarter": quarter, "clock": clock, "team": "",
        "play_text": "", "seq": 0,
    }


def _rams_rows():
    return [
        _row("401", "stafford", {"pass_td": 1}, "2026-10-25T20:48:31Z", "1", "6:25"),
        _row("401", "allen-d", {"rec_td": 1}, "2026-10-25T20:48:31Z", "1", "6:25"),
        _row("402", "stafford", {"pass_td": 1}, "2026-10-25T21:13:42Z", "2", "10:01"),
        _row("402", "allen-d", {"rec_td": 1}, "2026-10-25T21:13:42Z", "2", "10:01"),
        _row("403", "stafford", {"pass_td": 1}, "2026-10-25T21:28:51Z", "2", "4:15"),
        _row("403", "nacua", {"rec_td": 1}, "2026-10-25T21:28:51Z", "2", "4:15"),
        _row("404", "kyren", {"rush_td": 1}, "2026-10-25T21:45:49Z", "2", "1:19"),
        _row("405", "cousins", {"pass_td": 1}, "2026-10-25T22:51:27Z", "4", "14:51"),
        _row("405", "tucker", {"rec_td": 1}, "2026-10-25T22:51:27Z", "4", "14:51"),
        _row("406", "cousins", {"pass_td": 1}, "2026-10-25T22:59:56Z", "4", "13:12"),
        _row("406", "jeanty", {"rec_td": 1}, "2026-10-25T22:59:56Z", "4", "13:12"),
        _row("407", "corum", {"rush_td": 1}, "2026-10-25T23:13:04Z", "4", "6:34"),
    ]


def _tnf_rows():
    return [
        _row("501", "love", {"pass_td": 1}, "2026-10-30T01:31:48Z", "2", "1:46"),
        _row("501", "golden", {"rec_td": 1}, "2026-10-30T01:31:48Z", "2", "1:46"),
    ]


def _wanted():
    return (
        clips.build_wanted_td_plays(
            RAMS_GAME, _rams_rows(), PID_TO_NAME, away="LAR", home="LV")
        + clips.build_wanted_td_plays(
            TNF_GAME, _tnf_rows(), PID_TO_NAME, away="CAR", home="GB")
    )


def _tweet(tid, text, created_at, author="NFL"):
    return {"id": tid, "text": text, "created_at": created_at, "author": author}


BACKTEST_TWEETS = [
    _tweet("t1", "Another TE touchdown on National Tight Ends Day. "
                 "Matthew Stafford connects with Davis Allen for a Rams TD!",
           "2026-10-25T20:50:10Z"),
    _tweet("t2", "Make that TWO touchdowns for Davis Allen on "
                 "National Tight Ends Day.", "2026-10-25T21:16:00Z"),
    _tweet("t3", "Puka Nacua adds six more to the board for the Rams.",
           "2026-10-25T21:30:40Z"),
    _tweet("2114474206234918976", "Kyren Williams extends the lead for the Rams.",
           "2026-10-25T21:48:30Z"),
    _tweet("2114490475415748672",
           "Tre Tucker with a smooth move on his way to the end zone.",
           "2026-10-25T22:53:09Z"),
    _tweet("2114492709000102117",
           "Kirk Cousins floats it to Ashton Jeanty for the TD.",
           "2026-10-25T23:02:02Z"),
    _tweet("2114496046902751503",
           "Blake Corum takes it into the end zone for 6.",
           "2026-10-25T23:15:18Z"),
    _tweet("2115980579091157136",
           "Jordan Love connects with Matthew Golden for the TD "
           "CARvsGB on Prime Video Stream on @NFLPlus https://t.co/NVabs2hOxv",
           "2026-10-30T01:34:18Z"),
]

EXPECTED = {
    "401": "t1", "402": "t2", "403": "t3",
    "404": "2114474206234918976", "405": "2114490475415748672",
    "406": "2114492709000102117", "407": "2114496046902751503",
    "501": "2115980579091157136",
}


# ---------------------------------------------------------------------------
# Wanted list + gate: the backtest plays all match
# ---------------------------------------------------------------------------

def test_wanted_plays_built_from_stored_rows():
    wanted = _wanted()
    assert {w["play_id"] for w in wanted} == set(EXPECTED)
    allen = next(w for w in wanted if w["play_id"] == "401")
    assert allen["scorers"] == ["Davis Allen"]
    assert allen["wallclock"] > 0
    assert "RamsNFL" in allen["accounts"] and "Raiders" in allen["accounts"]


def test_all_backtest_captions_match_their_plays():
    matches = clips.match_tweets(BACKTEST_TWEETS, _wanted())
    got = {m["play_id"]: m["tweet_id"] for m in matches}
    assert got == EXPECTED
    kyren = next(m for m in matches if m["play_id"] == "404")
    assert kyren["scorer"] == "Kyren Williams"
    assert kyren["tweet_url"] == (
        "https://x.com/NFL/status/2114474206234918976")


def test_second_allen_td_disambiguated_by_window():
    # The "TWO touchdowns" caption must land on the second Allen TD only.
    matches = clips.match_tweets([BACKTEST_TWEETS[1]], _wanted())
    assert [m["play_id"] for m in matches] == ["402"]


def test_team_account_tweet_matches():
    matches = clips.match_tweets(
        [_tweet("t9", "Puka Nacua adds six more to the board for the Rams.",
                "2026-10-25T21:30:40Z", author="RamsNFL")],
        _wanted(),
    )
    assert [m["play_id"] for m in matches] == ["403"]


# ---------------------------------------------------------------------------
# Gate negatives
# ---------------------------------------------------------------------------

def test_generic_caption_naming_nobody_does_not_match():
    matches = clips.match_tweets(
        [_tweet("n1", "What a finish in Las Vegas.", "2026-10-25T21:48:30Z")],
        _wanted(),
    )
    assert matches == []


def test_roundup_caption_rejected_even_when_it_names_the_scorer():
    matches = clips.match_tweets(
        [_tweet("n2", "Every touchdown from a wild one in Las Vegas: "
                      "Davis Allen, Kyren Williams and all the highlights.",
                "2026-10-25T21:48:30Z")],
        _wanted(),
    )
    assert matches == []


def test_tweet_outside_window_does_not_match():
    matches = clips.match_tweets(
        [_tweet("n3", "Kyren Williams extends the lead for the Rams.",
                "2026-10-25T22:05:49Z")],  # 20 minutes after the play
        _wanted(),
    )
    assert matches == []


def test_unrelated_account_does_not_match():
    matches = clips.match_tweets(
        [_tweet("n4", "Kyren Williams extends the lead for the Rams.",
                "2026-10-25T21:48:30Z", author="AdamSchefter")],
        _wanted(),
    )
    assert matches == []


def test_caption_naming_other_scorer_outside_his_window_does_not_match():
    # Tucker's real caption text, posted during Kyren's window: Tucker's own
    # play is hours from its window, and Kyren is not named. No match.
    matches = clips.match_tweets(
        [_tweet("n5", "Tre Tucker with a smooth move on his way to the end zone.",
                "2026-10-25T21:48:30Z")],
        _wanted(),
    )
    assert matches == []


def test_surname_suffix_and_punctuation_normalization():
    assert clips.surname_of("Davis Allen Jr.") == "allen"
    assert clips.surname_of("Marvin Harrison Jr") == "harrison"
    assert clips.surname_of("Amon-Ra St. Brown") == "brown"
    assert clips.surname_of("Jaxon Smith-Njigba") == "njigba"


# ---------------------------------------------------------------------------
# Storage: idempotent upsert + reads, via a fake psycopg connection
# ---------------------------------------------------------------------------

class _Res:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeClipConn:
    """In-memory redzone_play_clips: enforces the (season, game, play) key."""

    def __init__(self):
        self.table = {}
        self.info = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=None):
        if "SELECT video_url FROM redzone_play_clips" in sql:
            key = (params[0], params[1], params[2])
            row = self.table.get(key)
            return _Res([(row["video_url"],)] if row else [])
        if "INSERT INTO redzone_play_clips" in sql:
            key = (params[0], params[1], params[2])
            if key in self.table:
                return _Res([])
            self.table[key] = {
                "game_id": params[1], "play_id": params[2],
                "tweet_id": params[3], "tweet_url": params[4],
                "author": params[5], "scorer": params[6], "ts": 1761438000.0,
                "video_url": params[9] if len(params) > 9 else "",
                "poster_url": params[10] if len(params) > 10 else "",
            }
            return _Res([{"play_id": params[2]}])
        if "UPDATE redzone_play_clips" in sql:
            if "replace-graphic" in sql:
                key = (params[8], params[9], params[10])
                row = self.table.get(key)
                if row is not None and row.get("video_url"):
                    row.update({
                        "video_url": params[0], "poster_url": params[1],
                        "tweet_id": params[2], "tweet_url": params[3],
                        "author": params[4], "scorer": params[5],
                    })
                    return _Res([{"play_id": params[10]}])
                return _Res([])
            key = (params[2], params[3], params[4])
            row = self.table.get(key)
            if row is not None and not row.get("video_url"):
                row["video_url"] = params[0]
                row["poster_url"] = params[1]
                return _Res([{"play_id": params[4]}])
            return _Res([])
        if "FROM redzone_play_clips" in sql:
            gids = set(params[1])
            return _Res([
                dict(row) for row in self.table.values()
                if row["game_id"] in gids
            ])
        return _Res([])


@pytest.fixture
def fake_clips_db(monkeypatch):
    conn = _FakeClipConn()
    monkeypatch.setattr(
        "dashboard_services.db.get_conn", lambda: conn)
    store._ENSURED_CLIP_TABLES.clear()
    return conn


def test_clip_upsert_is_idempotent_and_first_match_wins(fake_clips_db):
    clip = {"tweet_id": "111", "tweet_url": "https://x.com/NFL/status/111",
            "author": "NFL", "scorer": "Kyren Williams", "tweet_ts": 1761438000.0}
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", clip) is True
    other = dict(clip, tweet_id="222", tweet_url="https://x.com/NFL/status/222")
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", other) is False
    got = store.get_play_clips(2026, [RAMS_GAME])
    assert got[(RAMS_GAME, "404")]["tweet_id"] == "111"


def test_clip_upsert_fills_media_on_existing_row(fake_clips_db):
    clip = {"tweet_id": "111", "tweet_url": "https://x.com/NFL/status/111",
            "author": "NFL", "scorer": "Kyren Williams", "tweet_ts": 1761438000.0}
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", clip) is True
    assert store.get_play_clips(2026, [RAMS_GAME])[(RAMS_GAME, "404")]["video_url"] == ""
    with_media = dict(
        clip,
        video_url="https://video.twimg.com/amplify_video/1/vid/avc1/1280x720/x.mp4",
        poster_url="https://pbs.twimg.com/amplify_video_thumb/1/img/x.jpg",
    )
    # Same play, media arrives later (backfill re-run): fills, and the
    # tweet identity still cannot change.
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", with_media) is True
    got = store.get_play_clips(2026, [RAMS_GAME])[(RAMS_GAME, "404")]
    assert got["tweet_id"] == "111"
    assert got["video_url"].endswith("1280x720/x.mp4")
    assert got["poster_url"].endswith("img/x.jpg")
    # Once media is filled, further upserts are no-ops again.
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", with_media) is False


# Real response shape captured from GET /2/tweets/:id with media
# expansions on an @NFL TD post (2026-11-01): one video, HLS + 4 MP4s.
_MEDIA_PAYLOAD = {
    "data": [{
        "id": "900", "text": "Flournoy TD", "created_at": "2026-11-01T19:01:00Z",
        "attachments": {"media_keys": ["16_1"]},
    }],
    "includes": {"media": [{
        "media_key": "16_1", "type": "video",
        "preview_image_url": "https://pbs.twimg.com/amplify_video_thumb/1/img/p.jpg",
        "variants": [
            {"content_type": "application/x-mpegURL", "url": "https://video.twimg.com/x/pl.m3u8"},
            {"bit_rate": 256000, "content_type": "video/mp4", "url": "https://video.twimg.com/x/480x270/a.mp4"},
            {"bit_rate": 2176000, "content_type": "video/mp4", "url": "https://video.twimg.com/x/1280x720/b.mp4"},
            {"bit_rate": 10368000, "content_type": "video/mp4", "url": "https://video.twimg.com/x/1920x1080/c.mp4"},
            {"bit_rate": 832000, "content_type": "video/mp4", "url": "https://video.twimg.com/x/640x360/d.mp4"},
        ],
    }]},
}


def test_fetcher_extracts_phone_sized_mp4_and_poster():
    def _http(url, params, bearer):
        if "/users/by/username/" in url:
            return {"data": {"id": "44196397"}}
        assert params["expansions"] == "attachments.media_keys"
        return _MEDIA_PAYLOAD

    fetcher = clips.XClipFetcher("test-bearer", http_get=_http)
    out = fetcher.user_tweets("NFL")
    assert len(out) == 1
    # Largest MP4 at or under the 2.5 Mbps cap: the 720p variant.
    assert out[0]["video_url"] == "https://video.twimg.com/x/1280x720/b.mp4"
    assert out[0]["poster_url"] == "https://pbs.twimg.com/amplify_video_thumb/1/img/p.jpg"


def test_match_output_carries_media():
    tweets = [dict(BACKTEST_TWEETS[0], video_url="https://video.twimg.com/v.mp4",
                   poster_url="https://pbs.twimg.com/p.jpg")]
    matches = clips.match_tweets(tweets, _wanted())
    assert matches
    assert matches[0]["video_url"] == "https://video.twimg.com/v.mp4"
    assert matches[0]["poster_url"] == "https://pbs.twimg.com/p.jpg"


def test_attach_helpers_stamp_only_matched_td_plays(fake_clips_db):
    store.upsert_play_clip(2026, RAMS_GAME, "404", {
        "tweet_id": "2114474206234918976",
        "tweet_url": "https://x.com/NFL/status/2114474206234918976",
        "author": "NFL", "scorer": "Kyren Williams",
        "tweet_ts": 1761438000.0,
    })
    clips_map = store.get_play_clips(2026, [RAMS_GAME])
    plays_by_game = {RAMS_GAME: [
        {"play_id": "404", "is_td": True, "pid": "kyren"},
        {"play_id": "403", "is_td": True, "pid": "nacua"},
        {"play_id": "404", "is_td": False, "pid": "stafford"},
    ]}
    n = store.attach_clips_to_plays(plays_by_game, clips_map)
    rows = plays_by_game[RAMS_GAME]
    assert n == 1
    assert rows[0]["clip_tweet_id"] == "2114474206234918976"
    assert "clip_tweet_id" not in rows[1]  # unmatched TD: payload unchanged
    assert "clip_tweet_id" not in rows[2]  # non-TD row of the same play

    moments = [
        {"kind": "td", "game_id": RAMS_GAME, "play_id": "404"},
        {"kind": "td", "game_id": RAMS_GAME, "play_id": "403"},
        {"kind": "big_gain", "game_id": RAMS_GAME, "play_id": "404"},
    ]
    assert store.attach_clips_to_moments(moments, clips_map) == 1
    assert moments[0]["clip_url"].endswith("2114474206234918976")
    assert "clip_url" not in moments[1]
    assert "clip_url" not in moments[2]


# ---------------------------------------------------------------------------
# Real shape: ESPN summary payload -> extracted rows -> wanted -> match
# ---------------------------------------------------------------------------

def test_real_espn_summary_shape_flows_to_a_match():
    from utils.scorezone import _iso_to_epoch, extract_espn_pbp_plays

    payload = {"drives": {"previous": [{
        "team": {"abbreviation": "LA"},
        "plays": [{
            "id": "401873021401",
            "sequenceNumber": "57",
            "text": "(Shotgun) M.Stafford pass short left to D.Allen "
                    "for 3 yards, TOUCHDOWN.",
            "clock": {"displayValue": "6:25"},
            "period": {"number": 1},
            "start": {"down": 2, "distance": 3,
                      "possessionText": "LV 3"},
            "end": {"possessionText": "LV 0"},
            "type": {"text": "Pass"},
            "scoringPlay": True,
            "wallclock": "2026-10-25T20:48:31Z",
            "homeScore": 0, "awayScore": 7,
        }],
    }]}}
    rows = extract_espn_pbp_plays(
        payload, RAMS_GAME,
        name_to_pid={"matthew stafford": "stafford",
                     "davis allen": "allen-d"},
    )
    td_rows = [r for r in rows if r.get("is_td")]
    assert td_rows, "extractor must flag the TD rows"
    assert all(r.get("wallclock") == "2026-10-25T20:48:31Z" for r in td_rows)

    wanted = clips.build_wanted_td_plays(
        RAMS_GAME, rows,
        {"stafford": "Matthew Stafford", "allen-d": "Davis Allen"},
        away="LAR", home="LV",
    )
    assert len(wanted) == 1
    assert "Davis Allen" in wanted[0]["scorers"]
    assert wanted[0]["wallclock"] == _iso_to_epoch("2026-10-25T20:48:31Z")

    matches = clips.match_tweets([BACKTEST_TWEETS[0]], wanted)
    assert [m["play_id"] for m in matches] == ["401873021401"]


# ---------------------------------------------------------------------------
# Pass orchestration with a fake fetcher (no network, no DB)
# ---------------------------------------------------------------------------

def test_clip_pass_matches_and_stores(monkeypatch):
    rows = _rams_rows()
    stored = []
    monkeypatch.setattr(store, "get_plays",
                        lambda season, gids: {RAMS_GAME: rows})
    monkeypatch.setattr(store, "get_play_clips", lambda season, gids: {})
    monkeypatch.setattr(
        store, "upsert_play_clip",
        lambda season, gid, pid, clip: stored.append((gid, pid, clip)) or True)

    class _FakeFetcher:
        def __init__(self):
            self.user_ids = {}

        def user_tweets(self, username, since_id=None, max_results=25):
            if username == "NFL":
                return [BACKTEST_TWEETS[3]]  # Kyren clip only
            return []

    from utils.scorezone import _iso_to_epoch

    now = _iso_to_epoch("2026-10-25T21:50:00Z")
    stats = clips.maybe_run_clip_pass(
        2026,
        [{"game_id": RAMS_GAME, "live": True, "final": False}],
        {pid: {"full_name": name, "position": "WR"}
         for pid, name in PID_TO_NAME.items()},
        now=float(now), fetcher=_FakeFetcher(),
    )
    assert stats["matched"] == 1
    assert stored and stored[0][1] == "404"
    assert stored[0][2]["tweet_id"] == "2114474206234918976"


def test_clip_pass_gated_without_token(monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    clips._last_pass_ts = 0.0
    stats = clips.maybe_run_clip_pass(
        2026, [{"game_id": RAMS_GAME, "live": True, "final": False}], {})
    assert stats == {}


# ---------------------------------------------------------------------------
# Backfill: windowed fetch, date discovery, orchestration, dry-run
# ---------------------------------------------------------------------------

def test_windowed_fetch_sends_time_range_and_follows_pagination():
    from utils.scorezone import _iso_to_epoch

    calls = []

    class _WindowHttp:
        def __call__(self, url, params, bearer):
            calls.append((url, dict(params)))
            if "/users/by/username/" in url:
                return {"data": {"id": "44196397"}}
            if "pagination_token" not in params:
                return {
                    "data": [{"id": "100", "text": "first",
                              "created_at": "2026-10-30T01:20:00Z"}],
                    "meta": {"next_token": "tok2"},
                }
            return {
                "data": [{"id": "101", "text": "second",
                          "created_at": "2026-10-30T01:34:18Z"}],
                "meta": {},
            }

    fetcher = clips.XClipFetcher("test-bearer", http_get=_WindowHttp())
    out = fetcher.user_tweets_window(
        "NFL",
        _iso_to_epoch("2026-10-30T01:16:48Z"),
        _iso_to_epoch("2026-10-30T01:46:48Z"),
    )
    assert [t["id"] for t in out] == ["100", "101"]
    assert all(t["author"] == "NFL" for t in out)
    tweet_calls = [p for u, p in calls if u.endswith("/tweets")]
    assert len(tweet_calls) == 2
    assert tweet_calls[0]["start_time"] == "2026-10-30T01:16:48Z"
    assert tweet_calls[0]["end_time"] == "2026-10-30T01:46:48Z"
    assert tweet_calls[0]["max_results"] == 100
    assert "pagination_token" not in tweet_calls[0]
    assert tweet_calls[1]["pagination_token"] == "tok2"
    assert tweet_calls[1]["start_time"] == "2026-10-30T01:16:48Z"


def test_games_final_on_date_filters_to_finals():
    payload = {"events": [
        {"date": "2026-10-30T00:15:00Z",
         "status": {"type": {"state": "post", "completed": True}},
         "competitions": [{"competitors": [
             {"homeAway": "away", "team": {"abbreviation": "CAR"},
              "score": "20"},
             {"homeAway": "home", "team": {"abbreviation": "GB"},
              "score": "27"},
         ]}]},
        {"date": "2026-10-29T17:00:00Z",
         "status": {"type": {"state": "pre", "completed": False}},
         "competitions": [{"competitors": [
             {"homeAway": "away", "team": {"abbreviation": "NYJ"},
              "score": ""},
             {"homeAway": "home", "team": {"abbreviation": "NE"},
              "score": ""},
         ]}]},
    ]}
    got = clips.games_final_on_date(
        "2026-10-29", http_get=lambda url, params: payload)
    # The extractor dates game ids from the UTC kickoff timestamp, so the
    # Thursday-night game (00:15Z Friday) carries 20261030. Ids stay
    # self-consistent: the payload joins on the same form everywhere.
    assert got == ["20261030_CAR@GB"]


def _tnf_summary_rows():
    """Play rows built from a real-shape ESPN summary payload (TNF)."""
    from utils.scorezone import extract_espn_pbp_plays

    payload = {"drives": {"previous": [{
        "team": {"abbreviation": "GB"},
        "plays": [{
            "id": "401873023384",
            "sequenceNumber": "84",
            "text": "(Shotgun) J.Love pass deep left to M.Golden "
                    "for 24 yards, TOUCHDOWN.",
            "clock": {"displayValue": "1:46"},
            "period": {"number": 2},
            "start": {"down": 1, "distance": 10,
                      "possessionText": "CAR 24"},
            "end": {"possessionText": "GB 0"},
            "type": {"text": "Pass"},
            "scoringPlay": True,
            "wallclock": "2026-10-30T01:31:48Z",
            "homeScore": 14, "awayScore": 7,
        }],
    }]}}
    return extract_espn_pbp_plays(
        payload, TNF_GAME,
        name_to_pid={"jordan love": "love", "matthew golden": "golden"},
    )


class _StatefulClipStore:
    def __init__(self):
        self.clips = {}
        self.upserts = []

    def get(self, season, gids):
        return dict(self.clips)

    def upsert(self, season, gid, play_id, clip):
        self.upserts.append((gid, play_id, clip))
        key = (gid, play_id)
        if key in self.clips:
            # Mirror prod: only an empty-media row can still change,
            # by gaining the media a later match carries.
            row = self.clips[key]
            if not row.get("video_url") and clip.get("video_url"):
                row["video_url"] = clip["video_url"]
                row["poster_url"] = clip.get("poster_url") or ""
                return True
            return False
        self.clips[key] = dict(clip)
        return True


class _WindowFetcher:
    def __init__(self, by_account):
        self.by_account = by_account
        self.calls = []
        self.user_ids = {}

    def user_tweets_window(self, username, start_ts, end_ts):
        self.calls.append((username, start_ts, end_ts))
        return list(self.by_account.get(username) or [])


def _backfill_setup(monkeypatch):
    store_fake = _StatefulClipStore()
    monkeypatch.setattr(store, "get_play_clips", store_fake.get)
    monkeypatch.setattr(store, "upsert_play_clip", store_fake.upsert)
    fetcher = _WindowFetcher({"NFL": [dict(
        BACKTEST_TWEETS[-1],
        video_url="https://video.twimg.com/amplify_video/golden.mp4",
        poster_url="https://pbs.twimg.com/amplify_video_thumb/golden.jpg",
    )]})
    players = {
        "love": {"full_name": "Jordan Love", "position": "QB"},
        "golden": {"full_name": "Matthew Golden", "position": "WR"},
    }
    return store_fake, fetcher, players


def test_backfill_matches_tnf_golden_td_from_summary(monkeypatch):
    from utils.scorezone import _iso_to_epoch

    store_fake, fetcher, players = _backfill_setup(monkeypatch)
    stats = clips.run_clip_backfill(
        2026, [TNF_GAME], fetcher=fetcher, nfl_players=players,
        rows_for_game=lambda gid: _tnf_summary_rows(),
    )
    game = stats["games"][0]
    assert game["error"] == ""
    assert game["wanted"] == 1
    assert game["matched"] == 1
    assert game["stored"] == 1
    assert stats["read_cost_usd"] == round(game["tweets"] * 0.005, 2)
    assert store_fake.upserts[0][2]["tweet_id"] == "2115980579091157136"
    # Fetch window pads the play wallclock by 15 min on both sides.
    wallclock = _iso_to_epoch("2026-10-30T01:31:48Z")
    nfl_call = next(c for c in fetcher.calls if c[0] == "NFL")
    assert nfl_call[1] == wallclock - 900
    assert nfl_call[2] == wallclock + 900


def test_backfill_dry_run_writes_nothing(monkeypatch):
    store_fake, fetcher, players = _backfill_setup(monkeypatch)
    stats = clips.run_clip_backfill(
        2026, [TNF_GAME], fetcher=fetcher, nfl_players=players,
        dry_run=True, rows_for_game=lambda gid: _tnf_summary_rows(),
    )
    assert stats["games"][0]["matched"] == 1
    assert stats["games"][0]["stored"] == 0
    assert store_fake.upserts == []
    assert store_fake.clips == {}


def test_backfill_second_run_is_idempotent(monkeypatch):
    store_fake, fetcher, players = _backfill_setup(monkeypatch)
    kwargs = dict(fetcher=fetcher, nfl_players=players,
                  rows_for_game=lambda gid: _tnf_summary_rows())
    first = clips.run_clip_backfill(2026, [TNF_GAME], **kwargs)
    assert first["games"][0]["stored"] == 1
    calls_after_first = len(fetcher.calls)
    second = clips.run_clip_backfill(2026, [TNF_GAME], **kwargs)
    game = second["games"][0]
    assert game["wanted"] == 0  # already clipped: skipped before fetching
    assert game["tweets"] == 0
    assert game["stored"] == 0
    assert len(fetcher.calls) == calls_after_first
    assert len(store_fake.upserts) == 1


def test_backfill_upgrades_medialess_clip_then_goes_idle(monkeypatch):
    # A clip stored before media capture (tweet read without video) is
    # re-wanted once, gains its MP4/poster, and then skips like any
    # complete clip. This is how the Thursday clips already in prod pick
    # up the native player.
    store_fake, fetcher, players = _backfill_setup(monkeypatch)
    bare = _WindowFetcher({"NFL": [BACKTEST_TWEETS[-1]]})
    kwargs = dict(nfl_players=players,
                  rows_for_game=lambda gid: _tnf_summary_rows())
    first = clips.run_clip_backfill(2026, [TNF_GAME], fetcher=bare, **kwargs)
    assert first["games"][0]["stored"] == 1
    key = next(iter(store_fake.clips))
    assert not store_fake.clips[key].get("video_url")
    second = clips.run_clip_backfill(2026, [TNF_GAME], fetcher=fetcher, **kwargs)
    assert second["games"][0]["wanted"] == 1
    assert second["games"][0]["stored"] == 1  # the media fill counts
    assert store_fake.clips[key]["video_url"].endswith("golden.mp4")
    third = clips.run_clip_backfill(2026, [TNF_GAME], fetcher=fetcher, **kwargs)
    assert third["games"][0]["wanted"] == 0
    assert third["games"][0]["stored"] == 0


def test_backfill_script_requires_bearer_token(monkeypatch):
    import importlib.util
    from pathlib import Path

    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    path = (Path(__file__).resolve().parent.parent
            / "scripts" / "backfill_scorezone_clips.py")
    spec = importlib.util.spec_from_file_location(
        "backfill_scorezone_clips", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--game", TNF_GAME]) == 2


def test_clip_stream_url_transform():
    raw = "https://video.twimg.com/amplify_video/123/vid/avc1/1280x720/x.mp4"
    out = clips.clip_stream_url(raw)
    assert out.startswith("/api/clip-stream?u=")
    import urllib.parse as _up
    assert _up.unquote(out.split("?u=", 1)[1]) == raw
    # Non-twimg and empty values pass through untouched.
    assert clips.clip_stream_url("https://example.com/v.mp4") == "https://example.com/v.mp4"
    assert clips.clip_stream_url("") == ""


def test_attach_stamps_proxied_stream_url(fake_clips_db):
    store.upsert_play_clip(2026, RAMS_GAME, "404", {
        "tweet_id": "2114474206234918976",
        "tweet_url": "https://x.com/NFL/status/2114474206234918976",
        "author": "NFL", "scorer": "Kyren Williams",
        "tweet_ts": 1761438000.0,
        "video_url": "https://video.twimg.com/amplify_video/9/vid/avc1/1280x720/a.mp4",
        "poster_url": "https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg",
    })
    clips_map = store.get_play_clips(2026, [RAMS_GAME])
    plays_by_game = {RAMS_GAME: [{"play_id": "404", "is_td": True, "pid": "kyren"}]}
    assert store.attach_clips_to_plays(plays_by_game, clips_map) == 1
    row = plays_by_game[RAMS_GAME][0]
    # The store keeps the raw twimg URL; the payload carries the
    # same-origin stream URL (twimg 403s foreign-Referer hotlinks).
    assert row["clip_video_url"].startswith("/api/clip-stream?u=https%3A%2F%2Fvideo.twimg.com")
    assert row["clip_poster_url"] == "https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg"


# ---------------------------------------------------------------------------
# Graphic posts are not clips: team accounts post square "TOUCHDOWN"
# graphics seconds after a score; the broadcast highlight is landscape.
# ---------------------------------------------------------------------------

def _media_entry(variant_dims, mtype="video"):
    return {
        "type": mtype,
        "preview_image_url": "https://pbs.twimg.com/thumb/p.jpg",
        "variants": [
            {"content_type": "video/mp4", "bit_rate": br,
             "url": f"https://video.twimg.com/amplify_video/9/vid/avc1/{d}/x{br}.mp4"}
            for br, d in variant_dims
        ],
    }


def test_best_video_rejects_square_graphic():
    media = {"k1": _media_entry([(432000, "320x320"), (1280000, "720x720")])}
    tweet = {"attachments": {"media_keys": ["k1"]}}
    out = clips._best_video(tweet, media)
    assert out.get("graphic_media") is True
    assert "video_url" not in out


def test_best_video_accepts_landscape_broadcast():
    media = {"k1": _media_entry([(832000, "640x360"), (2176000, "1280x720")])}
    tweet = {"attachments": {"media_keys": ["k1"]}}
    out = clips._best_video(tweet, media)
    assert out["video_url"].endswith("1280x720/x2176000.mp4")
    assert not out.get("graphic_media")


def test_video_url_qualifies():
    assert clips.video_url_qualifies(
        "https://video.twimg.com/amplify_video/1/vid/avc1/1280x720/a.mp4")
    assert not clips.video_url_qualifies(
        "https://video.twimg.com/amplify_video/1/vid/avc1/720x720/a.mp4")
    assert not clips.video_url_qualifies(
        "https://video.twimg.com/amplify_video/1/vid/avc1/720x1280/a.mp4")
    assert clips.video_url_qualifies("https://video.twimg.com/plain.mp4")
    assert not clips.video_url_qualifies("")


def test_graphic_tweet_never_matches():
    graphic = dict(BACKTEST_TWEETS[0], graphic_media=True)
    matches = clips.match_tweets([graphic], _wanted())
    assert matches == []
    real = dict(BACKTEST_TWEETS[0],
                video_url="https://video.twimg.com/amplify_video/9/vid/avc1/1280x720/a.mp4")
    matches = clips.match_tweets([graphic, real], _wanted())
    assert len(matches) == 1
    assert matches[0]["video_url"].endswith("1280x720/a.mp4")


def test_upsert_replaces_graphic_row_with_broadcast(fake_clips_db):
    graphic = {
        "tweet_id": "900", "tweet_url": "https://x.com/Chiefs/status/900",
        "author": "Chiefs", "scorer": "Rashee Rice", "tweet_ts": 1761438000.0,
        "video_url": "https://video.twimg.com/amplify_video/9/vid/avc1/720x720/g.mp4",
        "poster_url": "https://pbs.twimg.com/thumb/g.jpg",
    }
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", graphic) is True
    broadcast = {
        "tweet_id": "901", "tweet_url": "https://x.com/NFL/status/901",
        "author": "NFL", "scorer": "Rashee Rice", "tweet_ts": 1761438120.0,
        "video_url": "https://video.twimg.com/amplify_video/9/vid/avc1/1280x720/b.mp4",
        "poster_url": "https://pbs.twimg.com/thumb/b.jpg",
    }
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", broadcast) is True
    got = store.get_play_clips(2026, [RAMS_GAME])[(RAMS_GAME, "404")]
    assert got["tweet_id"] == "901"
    assert got["video_url"].endswith("1280x720/b.mp4")
    # A qualifying row is never replaced by another qualifying match.
    other = dict(broadcast, tweet_id="902",
                 video_url="https://video.twimg.com/amplify_video/9/vid/avc1/1280x720/c.mp4")
    assert store.upsert_play_clip(2026, RAMS_GAME, "404", other) is False
