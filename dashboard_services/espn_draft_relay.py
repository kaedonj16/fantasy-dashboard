"""ESPN live-draft relay: tokens, snapshot store, bookmarklet helper.

Used by the browser extension (session) and the mobile bookmarklet / iOS
Shortcut (bearer token). Observe-only — never talks to ESPN and never
submits picks.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Draft-night window: long enough for a slow snake, short enough to rotate.
DEFAULT_TTL_SECONDS = 12 * 60 * 60
MAX_TTL_SECONDS = 24 * 60 * 60
_STORE_LOCK = threading.Lock()
_STORE: Dict[str, Dict[str, Any]] = {}


def _secret() -> bytes:
    return (
        os.environ.get("ESPN_RELAY_SECRET")
        or os.environ.get("FLASK_SECRET_KEY")
        or "br-fantasy-espn-relay"
    ).encode()


def _store_key(league_id: str, season: int) -> str:
    return f"espn_relay:{str(league_id).strip()}:{int(season)}"


def mint_relay_token(
    *,
    league_id: str,
    season: int,
    account_id: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Dict[str, Any]:
    """Return a signed token bound to one ESPN league/season."""
    lid = str(league_id or "").strip()
    if not lid:
        raise ValueError("league_id required")
    try:
        season_i = int(season)
    except (TypeError, ValueError) as exc:
        raise ValueError("season required") from exc
    ttl = max(300, min(int(ttl_seconds or DEFAULT_TTL_SECONDS), MAX_TTL_SECONDS))
    exp = int(time.time()) + ttl
    nonce = secrets.token_hex(8)
    aid = str(account_id or "").strip()
    payload = f"{lid}|{season_i}|{exp}|{nonce}|{aid}"
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:40]
    token = f"{lid}.{season_i}.{exp}.{nonce}.{aid or '0'}.{sig}"
    return {
        "token": token,
        "league_id": lid,
        "season": season_i,
        "expires_at": exp,
        "expires_in": ttl,
        "account_id": aid or None,
    }


def verify_relay_token(token: str) -> Optional[Dict[str, Any]]:
    """Return claims if the token is valid and unexpired, else None."""
    parts = str(token or "").strip().split(".")
    if len(parts) != 6:
        return None
    lid, season_s, exp_s, nonce, aid, sig = parts
    try:
        season_i = int(season_s)
        exp = int(exp_s)
    except (TypeError, ValueError):
        return None
    if exp < int(time.time()):
        return None
    if not lid or not nonce or not sig:
        return None
    aid_norm = "" if aid in ("0", "", "none", "null") else aid
    payload = f"{lid}|{season_i}|{exp}|{nonce}|{aid_norm}"
    expect = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:40]
    if not hmac.compare_digest(sig, expect):
        return None
    return {
        "league_id": lid,
        "season": season_i,
        "expires_at": exp,
        "account_id": aid_norm or None,
    }


def put_relay_snapshot(
    league_id: str,
    season: int,
    snapshot: Mapping[str, Any],
    *,
    source: str = "relay",
) -> Dict[str, Any]:
    """Store the latest normalized (or raw-ready) relay payload for a draft."""
    key = _store_key(league_id, season)
    entry = {
        "league_id": str(league_id),
        "season": int(season),
        "source": str(source or "relay"),
        "updated_at": int(time.time()),
        "payload": dict(snapshot),
    }
    with _STORE_LOCK:
        _STORE[key] = entry
    _redis_put(key, entry)
    return entry


def get_relay_snapshot(league_id: str, season: int) -> Optional[Dict[str, Any]]:
    key = _store_key(league_id, season)
    with _STORE_LOCK:
        local = _STORE.get(key)
    if local:
        return local
    return _redis_get(key)


def clear_relay_snapshot(league_id: str, season: int) -> None:
    key = _store_key(league_id, season)
    with _STORE_LOCK:
        _STORE.pop(key, None)
    _redis_delete(key)


def _redis_client():
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore
        return redis.from_url(url, socket_timeout=1.5, socket_connect_timeout=1.5)
    except Exception:
        return None


def _redis_put(key: str, entry: Mapping[str, Any]) -> None:
    client = _redis_client()
    if not client:
        return
    try:
        client.setex(key, DEFAULT_TTL_SECONDS, json.dumps(entry, separators=(",", ":")))
    except Exception as exc:
        logger.info("[espn-relay] redis put skipped error_type=%s", type(exc).__name__)


def _redis_get(key: str) -> Optional[Dict[str, Any]]:
    client = _redis_client()
    if not client:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        if isinstance(data, dict):
            with _STORE_LOCK:
                _STORE[key] = data
            return data
    except Exception as exc:
        logger.info("[espn-relay] redis get skipped error_type=%s", type(exc).__name__)
    return None


def _redis_delete(key: str) -> None:
    client = _redis_client()
    if not client:
        return
    try:
        client.delete(key)
    except Exception:
        pass


def site_origin(request_host_url: Optional[str] = None) -> str:
    env = (os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or "").rstrip("/")
    if env:
        return env
    if request_host_url:
        return str(request_host_url).rstrip("/")
    return "https://www.brfantasyfootball.com"


def build_bookmarklet(origin: str, token: str) -> str:
    """javascript: bookmark that syncs the open ESPN draft room to BR Fantasy."""
    compact = " ".join(_relay_script_body(origin, token, source="bookmarklet", for_ios_shortcut=False).split())
    return "javascript:" + compact


def shortcut_javascript(origin: str, token: str) -> str:
    """iOS Shortcuts 'Run JavaScript on Web Page' payload.

    Must call Apple's ``completion(result)`` on every exit path — pasting the
    bookmarklet (``javascript:…``) will fail with that exact error.
    """
    return " ".join(_relay_script_body(origin, token, source="ios-shortcut", for_ios_shortcut=True).split())


def _relay_script_body(origin: str, token: str, *, source: str, for_ios_shortcut: bool) -> str:
    """Shared ESPN draft → BR Fantasy relay script (bookmarklet + iOS Shortcut).

    Prefer React fiber ``draftDetail`` (live room truth). Fall back to a
    same-origin ESPN ``mDraftDetail`` fetch with cookies if fiber is empty —
    mid-draft REST is often stale, but better than nothing. On iOS Shortcuts,
    fiber discovery must use ``Object.getOwnPropertyNames`` (``for…in`` can miss
    ``__reactFiber$`` keys).
    """
    origin_js = json.dumps(str(origin).rstrip("/"))
    token_js = json.dumps(str(token))
    source_js = json.dumps(str(source))
    if for_ios_shortcut:
        finish_fn = (
            "function finish(m,ok){try{if(m)toast(m);}catch(e){}"
            "try{completion(ok?String(m||'ok'):null);}catch(e){}}"
        )
        early_open = "finish('Open your ESPN draft first.',false);return;"
        early_host = "finish('Open fantasy.espn.com draft, then run sync again.',false);return;"
        no_picks = (
            "finish('No picks found in page state. Open the Board or Pick History "
            "tab, wait a second, sync again — or use a laptop + Chrome extension.',false);"
        )
        sync_fail = "finish((res.b&&res.b.error)||'Sync failed',false);"
        sync_ok = (
            "finish('Synced '+(res.b.picks?res.b.picks.length:data.picks.length)"
            "+' picks to BR Fantasy',true);"
        )
        sync_net = "finish('Network error syncing to BR Fantasy',false);"
        toast_catch = "try{alert(m);}catch(e2){}"
    else:
        finish_fn = "function finish(m,ok){try{if(m)toast(m);}catch(e){}}"
        early_open = "return finish('Open your ESPN draft first.',false);"
        early_host = "return finish('Open fantasy.espn.com draft, then run sync again.',false);"
        no_picks = (
            "return finish('No picks found in page state. Open the Board or Pick History "
            "tab, wait a second, sync again — or use a laptop + Chrome extension.',false);"
        )
        sync_fail = "finish((res.b&&res.b.error)||'Sync failed',false);"
        sync_ok = (
            "finish('Synced '+(res.b.picks?res.b.picks.length:data.picks.length)"
            "+' picks to BR Fantasy',true);"
        )
        sync_net = "finish('Network error syncing to BR Fantasy',false);"
        toast_catch = "alert(m);"

    body = f"""
(function(){{
  var O={origin_js},T={token_js};
  function toast(m){{try{{var d=document.createElement('div');d.textContent=m;
  d.setAttribute('style','position:fixed;z-index:2147483647;left:12px;right:12px;bottom:16px;padding:12px 14px;border-radius:10px;background:#0f172a;color:#fff;font:600 13px/1.35 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)');
  document.documentElement.appendChild(d);setTimeout(function(){{d.remove();}},4200);}}catch(e){{{toast_catch}}}}}
  {finish_fn}
  function selectedPid(pid){{if(pid==null)return false;var s=String(pid);if(s===''||s==='0'||s==='-1'||s==='null'||s==='None')return false;var n=+s;if(!isNaN(n)&&(n===0||n===-1))return false;return true;}}
  function isPick(o){{return o&&typeof o==='object'&&(o.overallPickNumber!=null||o.overallPick!=null||o.pick_no!=null);}}
  function norm(r){{if(!isPick(r))return null;var pid=r.playerId!=null?r.playerId:r.player_id;if(!selectedPid(pid))return null;return{{overallPickNumber:Number(r.overallPickNumber||r.overallPick||r.pick_no),playerId:pid,teamId:r.teamId!=null?r.teamId:r.team_id,roundId:r.roundId!=null?Number(r.roundId):null,roundPickNumber:r.roundPickNumber!=null?Number(r.roundPickNumber):(r.roundPick!=null?Number(r.roundPick):null),keeper:!!(r.keeper||r.reservedForKeeper)}};}}
  function fromDetail(d){{if(!d||!Array.isArray(d.picks))return null;var picks=d.picks.map(norm).filter(Boolean);if(!picks.length)return null;return{{inProgress:d.inProgress===true,drafted:d.drafted===true,picks:picks}};}}
  function score(hit){{return hit&&hit.picks?hit.picks.length:0;}}
  function consider(best,hit){{if(!hit)return best;if(!best||score(hit)>score(best))return hit;return best;}}
  function walk(root){{var best=null,seen=typeof WeakSet!=='undefined'?new WeakSet():null,q=[root],n=0;
  while(q.length&&n<9000){{var c=q.shift();n++;if(!c||typeof c!=='object')continue;
  try{{if(seen){{if(seen.has(c))continue;seen.add(c);}}else{{if(c.__brSeen)continue;c.__brSeen=1;}}}}catch(e){{continue;}}
  try{{if(c.draftDetail)best=consider(best,fromDetail(c.draftDetail));}}catch(e){{}}
  try{{if(Array.isArray(c.picks)&&c.picks.length&&isPick(c.picks[0])){{var mapped=c.picks.map(norm).filter(Boolean);if(mapped.length)best=consider(best,{{inProgress:!!c.inProgress,drafted:!!c.drafted,picks:mapped}});}}}}catch(e){{}}
  var keys=['memoizedProps','pendingProps','stateNode','state','props','child','sibling','return','alternate','dependencies','memoizedState'];
  for(var i=0;i<keys.length;i++){{try{{if(c[keys[i]])q.push(c[keys[i]]);}}catch(e){{}}}}
  try{{var own=Object.getOwnPropertyNames(c);for(var j=0;j<Math.min(own.length,60);j++){{var k=own[j];if(k==='draftDetail'){{try{{best=consider(best,fromDetail(c[k]));}}catch(e){{}}}}try{{var v=c[k];if(v&&typeof v==='object')q.push(v);}}catch(e){{}}}}}}catch(e){{}}}}
  return best;}}
  function fiberKeys(el){{var out=[];try{{var names=Object.getOwnPropertyNames(el);for(var i=0;i<names.length;i++){{var k=names[i];if(k.indexOf('__reactFiber')===0||k.indexOf('__reactInternalInstance')===0||k.indexOf('__reactContainer')===0||k.indexOf('_reactRootContainer')===0)out.push(k);}}}}catch(e){{}}
  try{{for(var k2 in el){{if(k2&&(k2.indexOf('__reactFiber')===0||k2.indexOf('__reactInternalInstance')===0||k2.indexOf('__reactContainer')===0||k2.indexOf('_reactRootContainer')===0)&&out.indexOf(k2)<0)out.push(k2);}}}}catch(e){{}}
  return out;}}
  function scanReact(){{var best=null;var els=[];
  [document.getElementById('espn-aria-root'),document.getElementById('root'),document.getElementById('fitt-analytics'),document.querySelector('[data-reactroot]'),document.querySelector('#pane-main'),document.body].forEach(function(el){{if(el)els.push(el);}});
  try{{var nodes=document.querySelectorAll('div,main,section');for(var i=0;i<Math.min(nodes.length,80);i++){{var el=nodes[i];if(fiberKeys(el).length)els.push(el);}}}}catch(e){{}}
  for(var i=0;i<els.length;i++){{var el=els[i];var keys=fiberKeys(el);for(var j=0;j<keys.length;j++){{try{{var root=el[keys[j]];if(root&&root._internalRoot&&root._internalRoot.current)best=consider(best,walk(root._internalRoot.current));best=consider(best,walk(root));}}catch(e){{}}}}}}
  return best;}}
  function parseApiPayload(data){{if(!data)return null;if(Array.isArray(data)&&data[0])data=data[0];if(data.draftDetail)return fromDetail(data.draftDetail);if(data.league&&data.league.draftDetail)return fromDetail(data.league.draftDetail);return null;}}
  function apiScan(leagueId,season,done){{
    if(!leagueId||!season){{done(null);return;}}
    var urls=[
      'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/'+encodeURIComponent(season)+'/segments/0/leagues/'+encodeURIComponent(leagueId)+'?view=mDraftDetail&view=mSettings&view=mTeam',
      'https://fantasy.espn.com/apis/v3/games/ffl/seasons/'+encodeURIComponent(season)+'/segments/0/leagues/'+encodeURIComponent(leagueId)+'?view=mDraftDetail&view=mSettings&view=mTeam'
    ];
    var i=0;
    function next(){{
      if(i>=urls.length){{done(null);return;}}
      var url=urls[i++];
      fetch(url,{{method:'GET',credentials:'include',mode:'cors',headers:{{'Accept':'application/json'}}}})
        .then(function(r){{return r.ok?r.json():null;}})
        .then(function(data){{var hit=parseApiPayload(data);if(hit&&hit.picks&&hit.picks.length)done(hit);else next();}})
        .catch(function(){{next();}});
    }}
    next();
  }}
  function postRelay(leagueId,season,data){{
    fetch(O+'/api/draft/espn-relay',{{method:'POST',mode:'cors',credentials:'omit',headers:{{'Content-Type':'application/json','Authorization':'Bearer '+T}},body:JSON.stringify({{leagueId:leagueId,season:season,inProgress:data.inProgress!==false,drafted:!!data.drafted,picks:data.picks,source:{source_js}}})}})
    .then(function(r){{return r.json().then(function(b){{return{{ok:r.ok,b:b}};}});}})
    .then(function(res){{if(!res.ok){{{sync_fail}return;}}{sync_ok}}})
    .catch(function(){{{sync_net}}});
  }}
  var u;try{{u=new URL(location.href);}}catch(e){{{early_open}}}
  if(location.hostname.indexOf('espn.com')<0){{{early_host}}}
  var leagueId=(u.searchParams.get('leagueId')||'').trim();
  var season=(u.searchParams.get('seasonId')||u.searchParams.get('season')||'').trim();
  var data=scanReact();
  function go(hit){{
    if(!hit||!hit.picks||!hit.picks.length){{{no_picks}return;}}
    hit.picks.sort(function(a,b){{return a.overallPickNumber-b.overallPickNumber;}});
    postRelay(leagueId,season,hit);
  }}
  if(data&&data.picks&&data.picks.length){{go(data);return;}}
  apiScan(leagueId,season,go);
}})();
""".strip()
    return body


def merge_live_with_relay(
    live_payload: Mapping[str, Any],
    relay_entry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Prefer the longer/fresher pick list between REST live and stored relay."""
    out = dict(live_payload or {})
    if not relay_entry or not isinstance(relay_entry.get("payload"), Mapping):
        return out
    relay = dict(relay_entry["payload"])
    live_picks = live_payload.get("picks") if isinstance(live_payload.get("picks"), list) else []
    relay_picks = relay.get("picks") if isinstance(relay.get("picks"), list) else []

    def _count(picks: list) -> int:
        n = 0
        for p in picks:
            if not isinstance(p, Mapping):
                continue
            pid = p.get("player_id") or p.get("external_player_id")
            if pid in (None, "", "0", "-1"):
                continue
            n += 1
        return n

    live_n = _count(live_picks)
    relay_n = _count(relay_picks)
    if relay_n > live_n or (relay_n == live_n and relay_n > 0 and live_n == 0):
        out["picks"] = relay_picks
        out["picks_observed"] = True
        out["live_detail_present"] = True
        out["relay_source"] = relay.get("source") or relay_entry.get("source") or "relay"
        out["relay_updated_at"] = relay_entry.get("updated_at")
        if relay.get("status"):
            out["status"] = relay["status"]
        if relay.get("in_progress") is not None:
            out["in_progress"] = relay["in_progress"]
        if relay.get("fingerprint"):
            out["fingerprint"] = relay["fingerprint"]
        elif out.get("fingerprint"):
            out["fingerprint"] = str(out["fingerprint"]) + "|relay"
    elif relay_n > 0 and live_n > 0:
        # Annotate that relay is available even when REST leads.
        out["relay_source"] = relay_entry.get("source")
        out["relay_updated_at"] = relay_entry.get("updated_at")
        out["relay_pick_count"] = relay_n
    return out
