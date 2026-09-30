"""Value Share table sorting: rendered sort keys + the shipped JS comparator.

The render tests need flask/pandas (guarded, per repo convention); the
comparator test drives the real functions from static/app.js under node with
plain fake rows, so it runs in the lightweight shard too.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


class _ValuesConn:
    def __init__(self, values):
        self._values = values

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _sql):
        values = self._values

        class _Result:
            def fetchall(self):
                return [{"player_id": pid, "v": v} for pid, v in values.items()]

        return _Result()


def _shares_html(monkeypatch):
    pytest.importorskip("flask")
    pd = pytest.importorskip("pandas")
    import dashboard_services.db as db
    import app as appmod

    values = {"a1": 300.0, "c1": 200.0, "b1": 100.0}
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _ValuesConn(values))

    team_stats = pd.DataFrame([
        {"owner": "Bravo", "PF": 200.0},
        {"owner": "Alpha", "PF": 100.0},
        {"owner": "Charlie", "PF": 300.0},
    ])
    ctx = {
        "team_stats": team_stats,
        "rosters": [
            {"roster_id": 1, "players": ["a1"]},
            {"roster_id": 2, "players": ["c1"]},
            {"roster_id": 3, "players": ["b1"]},
        ],
        "roster_map": {"1": "Alpha", "2": "Charlie", "3": "Bravo"},
        "df_weekly": None,
        "roster_positions": [],
    }
    return appmod.render_share_rankings(ctx)


def test_value_share_table_renders_sortable_headers_and_row_keys(monkeypatch):
    html = _shares_html(monkeypatch)

    assert ('<table class="standings-shares-table" '
            'data-sort-key="value" data-sort-dir="desc">') in html
    for key in ("rank", "team", "value", "production"):
        assert f'data-share-sort="{key}"' in html
    assert 'aria-sort="descending" class="sorted-desc"' in html
    assert 'data-share-sort="production"' in html and "Production Share" in html

    rows = re.findall(
        r'data-share-rank="(\d+)" data-share-team="([^"]+)" '
        r'data-share-value="([\d.]+)" data-share-production="([\d.]+)"',
        html,
    )
    # Default order is value share descending: Alpha 50%, Charlie 33.3%,
    # Bravo 16.7%. Production (PF) deliberately orders differently.
    assert [r[1] for r in rows] == ["Alpha", "Charlie", "Bravo"]
    assert [r[0] for r in rows] == ["1", "2", "3"]
    by_team = {r[1]: r for r in rows}
    assert float(by_team["Alpha"][2]) == pytest.approx(50.0)
    assert float(by_team["Alpha"][3]) == pytest.approx(100.0 / 6.0)
    assert float(by_team["Charlie"][2]) == pytest.approx(100.0 / 3.0)
    assert float(by_team["Charlie"][3]) == pytest.approx(50.0)
    assert float(by_team["Bravo"][3]) == pytest.approx(100.0 / 3.0)


def test_value_share_sort_wiring_and_styles_are_shipped():
    assert "function standingsShareSortValue(row, key)" in APP_JS
    assert "function compareStandingsShareRows(a, b, key, dir)" in APP_JS
    assert "function initStandingsSharesSort(root = document)" in APP_JS
    # initPageRoot reaches the shares table through initStandingsSort, which
    # also covers week-selector swaps of just the shares panel.
    assert "initStandingsSharesSort(root);" in APP_JS
    assert 'closest?.("[data-share-sort]")' in APP_JS
    assert ".standings-shares-sort-btn" in CSS
    assert ".standings-shares-sort-btn:focus-visible" in CSS
    assert '.standings-shares-table .sorted-desc::after' in CSS


def _extract_fn(src, name):
    start = src.index(f"function {name}(")
    depth = 0
    for i in range(src.index("{", start), len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_value_share_comparator_orders_every_column():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    fns = "\n".join([
        _extract_fn(APP_JS, "standingsShareSortValue"),
        _extract_fn(APP_JS, "compareStandingsShareRows"),
    ])
    driver = fns + """
function row(rank, team, value, production) {
  const attrs = {
    "data-share-rank": String(rank),
    "data-share-team": team,
    "data-share-value": String(value),
    "data-share-production": String(production),
  };
  return { getAttribute: k => (k in attrs ? attrs[k] : null) };
}
const base = [
  row(1, "Alpha", 50, 100 / 6),
  row(2, "Charlie", 100 / 3, 50),
  row(3, "Bravo", 50 / 3, 100 / 3),
  row(4, "Delta", 100 / 3, 10),
  row(5, "Echo", 100 / 3, 20),
];
const names = rows => rows.map(r => r.getAttribute("data-share-team"));
const sorted = (key, dir) => names(base.slice().sort((a, b) =>
  compareStandingsShareRows(a, b, key, dir)));
console.log(JSON.stringify({
  valueDesc: sorted("value", -1),
  valueAsc: sorted("value", 1),
  productionDesc: sorted("production", -1),
  productionAsc: sorted("production", 1),
  teamAsc: sorted("team", 1),
  teamDesc: sorted("team", -1),
  rankAsc: sorted("rank", 1),
  rankDesc: sorted("rank", -1),
}));
"""
    res = subprocess.run([node, "-e", driver], capture_output=True, text=True,
                         timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])

    assert out["valueDesc"] == ["Alpha", "Charlie", "Delta", "Echo", "Bravo"]
    # Equal value shares keep their original rank order in both directions.
    assert out["valueAsc"] == ["Bravo", "Charlie", "Delta", "Echo", "Alpha"]
    assert out["productionDesc"] == ["Charlie", "Bravo", "Echo", "Alpha", "Delta"]
    assert out["productionAsc"] == ["Delta", "Alpha", "Echo", "Bravo", "Charlie"]
    assert out["teamAsc"] == ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    assert out["teamDesc"] == ["Echo", "Delta", "Charlie", "Bravo", "Alpha"]
    assert out["rankAsc"] == ["Alpha", "Charlie", "Bravo", "Delta", "Echo"]
    assert out["rankDesc"] == ["Echo", "Delta", "Bravo", "Charlie", "Alpha"]


def test_value_share_header_clicks_sort_and_renumber():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    fns = "\n".join([
        _extract_fn(APP_JS, "bindOnce"),
        _extract_fn(APP_JS, "standingsShareSortValue"),
        _extract_fn(APP_JS, "compareStandingsShareRows"),
        _extract_fn(APP_JS, "initStandingsSharesSort"),
    ])
    driver = fns + """
function matches(el, sel) {
  if (sel === "tr") return el.tagName === "TR";
  if (sel === "thead th") return el.tagName === "TH";
  if (sel === ".standings-shares-table") return el.classList.contains("standings-shares-table");
  if (sel === ".standings-shares-rk") return el.classList.contains("standings-shares-rk");
  if (sel === "[data-share-sort]") return el.getAttribute("data-share-sort") !== null;
  return false;
}
class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.attrs = {};
    this.children = [];
    this.parent = null;
    this.textContent = "";
    this._listeners = {};
    const classes = new Set();
    this.classList = {
      add: (...cs) => cs.forEach(c => classes.add(c)),
      remove: (...cs) => cs.forEach(c => classes.delete(c)),
      toggle: (c, f) => {
        if (f === undefined) f = !classes.has(c);
        if (f) classes.add(c); else classes.delete(c);
        return f;
      },
      contains: c => classes.has(c),
    };
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(type, handler) { this._listeners[type] = handler; }
  appendChild(c) { c.parent = this; this.children.push(c); return c; }
  replaceChildren(...cs) {
    this.children = [];
    cs.forEach(c => this.appendChild(c));
  }
  contains(el) {
    let n = el;
    while (n) { if (n === this) return true; n = n.parent; }
    return false;
  }
  closest(sel) {
    let n = this;
    while (n) { if (matches(n, sel)) return n; n = n.parent; }
    return null;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    const walk = n => n.children.forEach(c => {
      if (matches(c, sel)) out.push(c);
      walk(c);
    });
    walk(this);
    return out;
  }
}
function makeRow(rank, team, value, production) {
  const tr = new El("tr");
  tr.setAttribute("data-share-rank", rank);
  tr.setAttribute("data-share-team", team);
  tr.setAttribute("data-share-value", value);
  tr.setAttribute("data-share-production", production);
  const rk = new El("td");
  rk.classList.add("standings-shares-rk");
  rk.textContent = String(rank);
  tr.appendChild(rk);
  return tr;
}
const root = new El("div");
const tbl = new El("table");
tbl.classList.add("standings-shares-table");
tbl.setAttribute("data-sort-key", "value");
tbl.setAttribute("data-sort-dir", "desc");
const thead = new El("thead");
const headRow = new El("tr");
const btns = {};
["rank", "team", "value", "production"].forEach(key => {
  const th = new El("th");
  const btn = new El("button");
  btn.setAttribute("data-share-sort", key);
  th.appendChild(btn);
  headRow.appendChild(th);
  btns[key] = btn;
});
thead.appendChild(headRow);
const tbody = new El("tbody");
tbody.appendChild(makeRow(1, "Alpha", 50, 100 / 6));
tbody.appendChild(makeRow(2, "Charlie", 100 / 3, 50));
tbody.appendChild(makeRow(3, "Bravo", 50 / 3, 100 / 3));
tbl.appendChild(thead);
tbl.appendChild(tbody);
tbl.tHead = thead;
tbl.tBodies = [tbody];
root.appendChild(tbl);

initStandingsSharesSort(root);
const click = key => thead._listeners.click({ target: btns[key] });
const teams = () => tbody.children.map(r => r.getAttribute("data-share-team"));
const ranks = () => tbody.children.map(r =>
  r.querySelector(".standings-shares-rk").textContent);
const thFor = key => btns[key].parent;

click("production");
const afterProd = {
  teams: teams(), ranks: ranks(),
  key: tbl.getAttribute("data-sort-key"), dir: tbl.getAttribute("data-sort-dir"),
  prodAria: thFor("production").getAttribute("aria-sort"),
  valueAria: thFor("value").getAttribute("aria-sort"),
  prodDesc: thFor("production").classList.contains("sorted-desc"),
};
click("team");
const afterTeamAsc = teams();
click("team");
const afterTeamDesc = {
  teams: teams(),
  firstRowDataRank: tbody.children[0].getAttribute("data-share-rank"),
  firstRowShownRank: ranks()[0],
};
click("value");
const afterValue = teams();
console.log(JSON.stringify({
  afterProd, afterTeamAsc, afterTeamDesc, afterValue,
}));
"""
    res = subprocess.run([node, "-e", driver], capture_output=True, text=True,
                         timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])

    prod = out["afterProd"]
    assert prod["teams"] == ["Charlie", "Bravo", "Alpha"]
    assert prod["ranks"] == ["1", "2", "3"]
    assert prod["key"] == "production" and prod["dir"] == "desc"
    assert prod["prodAria"] == "descending" and prod["prodDesc"]
    assert prod["valueAria"] == "none"
    # New column starts ascending for Team; clicking again reverses it.
    assert out["afterTeamAsc"] == ["Alpha", "Bravo", "Charlie"]
    assert out["afterTeamDesc"]["teams"] == ["Charlie", "Bravo", "Alpha"]
    # The shown rank renumbers to the display position, while the row keeps
    # its original value rank in data-share-rank (Charlie was rank 2).
    assert out["afterTeamDesc"]["firstRowShownRank"] == "1"
    assert out["afterTeamDesc"]["firstRowDataRank"] == "2"
    # Value Share's first click from another column restores high-to-low.
    assert out["afterValue"] == ["Alpha", "Charlie", "Bravo"]
