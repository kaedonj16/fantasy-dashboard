"""Behavioral coverage for the bounded player-details warm-up layer."""
from pathlib import Path
import json
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODAL = (ROOT / "static/player_modal.js").read_text()
APP = (ROOT / "static/app.js").read_text()


def _loader_source():
    start = MODAL.index("// Shared, bounded player-details request layer")
    end = MODAL.index("\n\nfunction openPlayerModal", start)
    return MODAL[start:end]


def _node(body):
    program = f"""
global.window=global;
global.location={{pathname:'/sleeper/2026/league-a/players',search:''}};
global.navigator={{onLine:true}};
global.brLeagueType=()=> 'sf'; global.brLeagueSize=()=>12;
const storage=new Map(); global.localStorage={{getItem:k=>storage.has(k)?storage.get(k):null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k),key:i=>Array.from(storage.keys())[i]||null,get length(){{return storage.size}}}};
{_loader_source()}
(async()=>{{{body}}})().catch(e=>{{console.error(e);process.exit(1)}});
"""
    return subprocess.run(["node", "-e", program], text=True, capture_output=True)


def test_loader_deduplicates_context_and_preserves_def_ids():
    result = _node("""
let resolve; let calls=0;
global.fetch=(url,init)=>{calls++;return new Promise(r=>resolve=()=>r({ok:true,status:200,json:async()=>({name:'49ers',player_id:'SF'})}))};
const a=pmPlayerDetails.load('SF',{speculative:true});
const b=pmPlayerDetails.load('SF',{});
if(calls!==1)throw Error('duplicate request'); resolve(); await Promise.all([a,b]);
const d=pmPlayerDetails.descriptor('SF',{});
if(!d.key.includes('sf|12')||!d.url.includes('league_type=sf')||!d.url.includes('league_size=12'))throw Error('context omitted');
console.log(JSON.stringify(pmPlayerDetails.stats()));
""")
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    assert stats["counters"]["inflightReuse"] == 1
    assert stats["entries"] == 1


def test_speculative_skip_adopted_by_click_retries_foreground():
    result = _node("""
let n=0, release; global.fetch=(url,init)=>{n++;if(n===1)return new Promise(r=>release=()=>r({ok:true,status:204}));return Promise.resolve({ok:true,status:200,json:async()=>({name:'Player',player_id:'7'})})};
const warm=pmPlayerDetails.load('7',{speculative:true}).catch(e=>e);
const click=pmPlayerDetails.load('7',{}); release(); const data=await click; await warm;
if(n!==2||data.name!=='Player')throw Error('foreground fallback missing');
console.log(JSON.stringify(pmPlayerDetails.stats().counters));
""")
    assert result.returncode == 0, result.stderr
    counters = json.loads(result.stdout)
    assert counters["speculativeSkips"] == 1
    assert counters["foregroundStarts"] == 1


def test_cache_is_lru_bounded_and_invalidation_rejects_stale_fill():
    result = _node("""
let pending; global.fetch=(url)=>new Promise(r=>pending=r);
const stale=pmPlayerDetails.load('old',{});pmPlayerDetails.invalidate();pending({ok:true,status:200,json:async()=>({name:'Old',player_id:'old'})});await stale;
if(pmPlayerDetails.stats().entries!==0)throw Error('stale response repopulated');
global.fetch=async url=>({ok:true,status:200,json:async()=>({name:url,player_id:url})});
for(let i=0;i<30;i++)await pmPlayerDetails.load(String(i),{});
const st=pmPlayerDetails.stats();if(st.entries!==24||st.bytes>2*1024*1024)throw Error('unbounded cache');console.log('ok');
""")
    assert result.returncode == 0, result.stderr



def test_account_context_isolates_memory_and_persistent_entries():
    result = _node("""
let calls=0;global.fetch=async()=>({ok:true,status:200,json:async()=>({name:'Player',player_id:'9'})});
await pmPlayerDetails.load('9',{account:'account-a'});
await pmPlayerDetails.load('9',{account:'account-b'});
if(pmPlayerDetails.stats().entries!==2)throw Error('accounts shared memory cache');
const keys=Array.from(storage.keys());if(keys.length!==2||!keys.every(k=>k.includes('_pm_account=account-')))throw Error('accounts shared persistent cache');
console.log('ok');
""")
    assert result.returncode == 0, result.stderr


def test_scheduler_limits_and_navigation_cleanup_are_wired():
    assert "DWELL_MS=500, MAX_ATTEMPTS=6, MAX_QUEUE=12" in MODAL
    assert "threshold:[.25],rootMargin:'0px'" in MODAL
    assert "navigator.connection" in MODAL and "saveData" in MODAL
    assert "pmStopVisibilityWarmup" in APP
    assert "pmPlayerDetails.invalidate()" in APP
    assert "state.abort?.abort()" in MODAL
    assert "m.removedNodes.forEach(forget)" in MODAL
    assert "pointerout" in MODAL
    prefetch = MODAL[MODAL.index("function pmPrefetchTabs"):MODAL.index("// ── Weekly")]
    assert "pmSwitchTab(" not in prefetch


@pytest.mark.integration
def test_speculative_cold_league_skips_without_rebuild(monkeypatch):
    flask = pytest.importorskip("flask")
    import app as appmod
    appmod.app.config.update(TESTING=True)
    monkeypatch.setattr(appmod, "DASHBOARD_CACHE", {})
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a: pytest.fail("cold rebuild"))
    with appmod.app.test_client() as client:
        response = client.get("/api/player-details/4046?league_id=cold&platform=sleeper&season=2026", headers={"X-BR-Speculative":"player-details"})
    assert response.status_code == 204
