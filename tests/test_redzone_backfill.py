from contextlib import contextmanager


def test_repairs_latest_rows_without_overwriting_other_metrics(monkeypatch):
    import scripts.backfill_redzone_metrics as repair
    writes = []
    class Conn:
        def execute(self, sql, params=None):
            if params is not None:
                writes.append((sql, params))
            else:
                assert 'DISTINCT ON (season, player_id)' in sql
                assert 'as_of_date DESC' in sql
            return self
        def fetchall(self):
            return [dict(id=1, season=2025, player_id='wr'),
                    dict(id=2, season=2025, player_id='rb'),
                    dict(id=3, season=2025, player_id='missing')]
    @contextmanager
    def connection():
        yield Conn()
    monkeypatch.setattr(repair, 'get_conn', connection)
    monkeypatch.setattr(repair, 'init_advanced_metrics_db', lambda: None)
    weeks = []
    monkeypatch.setattr(repair, 'build_weekly_metrics', lambda season: weeks.append(season))
    monkeypatch.setattr(repair, 'build_usage_map_for_season', lambda *args: {
        'wr': dict(games=10, red_zone_available=True, rec_rz_tgt_pg=1.2, rush_rz_att_pg=0),
        'rb': dict(games=10, red_zone_available=True, rec_rz_tgt_pg=0, rush_rz_att_pg=0),
        'missing': dict(games=10, red_zone_available=False),
    })
    result = repair.backfill_redzone_metrics()
    assert result == dict(updated=2, unavailable=1)
    assert weeks == [2025]
    assert [p for _, p in writes] == [(1.2, 0, 1.2, 1), (0, 0, 0, 2)]
    assert all('games =' not in sql and 'target_share =' not in sql for sql, _ in writes)


def test_incremental_weekly_build_revisits_old_missing_rz(monkeypatch):
    import data_building.weekly_metrics as wm
    class Conn:
        def execute(self, sql, params=None):
            assert 'BOOL_OR(rz_targets IS NULL OR rz_carries IS NULL)' in sql
            return self
        def fetchall(self):
            return [dict(week=w, needs_rz=(w == 1)) for w in range(1, 19)]
    @contextmanager
    def connection():
        yield Conn()
    monkeypatch.setattr(wm, 'get_conn', connection)
    monkeypatch.setattr(wm, 'init_weekly_metrics_db', lambda: None)
    monkeypatch.setattr(wm, 'load_players_index', lambda: {})
    fetched = []
    monkeypatch.setattr(wm, 'fetch_week_stats', lambda season, week: fetched.append(week) or {})
    assert wm.build_weekly_metrics(2025) == 0
    assert fetched == [1, 17, 18]
