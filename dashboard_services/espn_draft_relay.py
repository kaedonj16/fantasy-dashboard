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
    """Shared ESPN draft → BR Fantasy relay script body."""
    origin_js = json.dumps(str(origin).rstrip("/"))
    token_js = json.dumps(str(token))
    source_js = json.dumps(str(source))

    # Build with explicit completion hooks for iOS Shortcuts.
    if for_ios_shortcut:
        body = f"""
(function(){{
  var O={origin_js},T={token_js};
  function toast(m){{try{{var d=document.createElement('div');d.textContent=m;
  d.setAttribute('style','position:fixed;z-index:2147483647;left:12px;right:12px;bottom:16px;padding:12px 14px;border-radius:10px;background:#0f172a;color:#fff;font:600 13px/1.35 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)');
  document.documentElement.appendChild(d);setTimeout(function(){{d.remove();}},3200);}}catch(e){{try{{alert(m);}}catch(e2){{}}}}}}
  function finish(m,ok){{try{{if(m)toast(m);}}catch(e){{}}try{{completion(ok?String(m||'ok'):null);}}catch(e){{}}}}
  function isPick(o){{return o&&typeof o==='object'&&(o.overallPickNumber!=null||o.overallPick!=null||o.pick_no!=null)&&(o.playerId!=null||o.player_id!=null||o.teamId!=null);}}
  function norm(r){{if(!isPick(r))return null;return{{overallPickNumber:Number(r.overallPickNumber||r.overallPick||r.pick_no),playerId:r.playerId!=null?r.playerId:r.player_id,teamId:r.teamId!=null?r.teamId:r.team_id,roundId:r.roundId!=null?Number(r.roundId):null,roundPickNumber:r.roundPickNumber!=null?Number(r.roundPickNumber):(r.roundPick!=null?Number(r.roundPick):null),keeper:!!(r.keeper||r.reservedForKeeper)}};}}
  function fromDetail(d){{if(!d||!Array.isArray(d.picks))return null;return{{inProgress:d.inProgress===true,drafted:d.drafted===true,picks:d.picks.map(norm).filter(Boolean)}};}}
  function walk(root){{var seen=new Set(),q=[root],n=0;while(q.length&&n<3500){{var c=q.shift();n++;if(!c||typeof c!=='object'||seen.has(c))continue;seen.add(c);
  if(c.draftDetail){{var x=fromDetail(c.draftDetail);if(x)return x;}}
  if(Array.isArray(c.picks)&&c.picks.length&&isPick(c.picks[0]))return{{inProgress:!!c.inProgress,drafted:!!c.drafted,picks:c.picks.map(norm).filter(Boolean)}};
  ['memoizedProps','pendingProps','stateNode','state','props','child','sibling','return'].forEach(function(k){{if(c[k])q.push(c[k]);}});
  try{{Object.keys(c).slice(0,30).forEach(function(k){{var v=c[k];if(v&&typeof v==='object'&&!seen.has(v))q.push(v);}});}}catch(e){{}}}}return null;}}
  function scan(){{var els=[document.getElementById('espn-aria-root'),document.getElementById('root'),document.body].filter(Boolean);for(var i=0;i<els.length;i++){{var el=els[i];for(var k in el){{if(k.indexOf('__reactFiber$')===0||k.indexOf('__reactInternalInstance$')===0){{var hit=walk(el[k]);if(hit)return hit;}}}}}}return null;}}
  var u;try{{u=new URL(location.href);}}catch(e){{finish('Open your ESPN draft first.',false);return;}}
  if(location.hostname.indexOf('espn.com')<0){{finish('Open fantasy.espn.com draft, then run sync again.',false);return;}}
  var leagueId=(u.searchParams.get('leagueId')||'').trim();
  var season=(u.searchParams.get('seasonId')||u.searchParams.get('season')||'').trim();
  var data=scan();
  if(!data||!data.picks||!data.picks.length){{finish('No picks found yet — wait for a pick, then tap sync again.',false);return;}}
  data.picks.sort(function(a,b){{return a.overallPickNumber-b.overallPickNumber;}});
  fetch(O+'/api/draft/espn-relay',{{method:'POST',mode:'cors',credentials:'omit',headers:{{'Content-Type':'application/json','Authorization':'Bearer '+T}},body:JSON.stringify({{leagueId:leagueId,season:season,inProgress:data.inProgress!==false,drafted:!!data.drafted,picks:data.picks,source:{source_js}}})}})
  .then(function(r){{return r.json().then(function(b){{return{{ok:r.ok,b:b}};}});}})
  .then(function(res){{if(!res.ok){{finish((res.b&&res.b.error)||'Sync failed',false);return;}}finish('Synced '+(res.b.picks?res.b.picks.length:data.picks.length)+' picks to BR Fantasy',true);}})
  .catch(function(){{finish('Network error syncing to BR Fantasy',false);}});
}})();
""".strip()
    else:
        body = f"""
(function(){{
  var O={origin_js},T={token_js};
  function toast(m){{try{{var d=document.createElement('div');d.textContent=m;
  d.setAttribute('style','position:fixed;z-index:2147483647;left:12px;right:12px;bottom:16px;padding:12px 14px;border-radius:10px;background:#0f172a;color:#fff;font:600 13px/1.35 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.35)');
  document.documentElement.appendChild(d);setTimeout(function(){{d.remove();}},3200);}}catch(e){{alert(m);}}}}
  function isPick(o){{return o&&typeof o==='object'&&(o.overallPickNumber!=null||o.overallPick!=null||o.pick_no!=null)&&(o.playerId!=null||o.player_id!=null||o.teamId!=null);}}
  function norm(r){{if(!isPick(r))return null;return{{overallPickNumber:Number(r.overallPickNumber||r.overallPick||r.pick_no),playerId:r.playerId!=null?r.playerId:r.player_id,teamId:r.teamId!=null?r.teamId:r.team_id,roundId:r.roundId!=null?Number(r.roundId):null,roundPickNumber:r.roundPickNumber!=null?Number(r.roundPickNumber):(r.roundPick!=null?Number(r.roundPick):null),keeper:!!(r.keeper||r.reservedForKeeper)}};}}
  function fromDetail(d){{if(!d||!Array.isArray(d.picks))return null;return{{inProgress:d.inProgress===true,drafted:d.drafted===true,picks:d.picks.map(norm).filter(Boolean)}};}}
  function walk(root){{var seen=new Set(),q=[root],n=0;while(q.length&&n<3500){{var c=q.shift();n++;if(!c||typeof c!=='object'||seen.has(c))continue;seen.add(c);
  if(c.draftDetail){{var x=fromDetail(c.draftDetail);if(x)return x;}}
  if(Array.isArray(c.picks)&&c.picks.length&&isPick(c.picks[0]))return{{inProgress:!!c.inProgress,drafted:!!c.drafted,picks:c.picks.map(norm).filter(Boolean)}};
  ['memoizedProps','pendingProps','stateNode','state','props','child','sibling','return'].forEach(function(k){{if(c[k])q.push(c[k]);}});
  try{{Object.keys(c).slice(0,30).forEach(function(k){{var v=c[k];if(v&&typeof v==='object'&&!seen.has(v))q.push(v);}});}}catch(e){{}}}}return null;}}
  function scan(){{var els=[document.getElementById('espn-aria-root'),document.getElementById('root'),document.body].filter(Boolean);for(var i=0;i<els.length;i++){{var el=els[i];for(var k in el){{if(k.indexOf('__reactFiber$')===0||k.indexOf('__reactInternalInstance$')===0){{var hit=walk(el[k]);if(hit)return hit;}}}}}}return null;}}
  var u;try{{u=new URL(location.href);}}catch(e){{return toast('Open your ESPN draft first.');}}
  if(location.hostname.indexOf('espn.com')<0){{return toast('Open fantasy.espn.com draft, then run sync again.');}}
  var leagueId=(u.searchParams.get('leagueId')||'').trim();
  var season=(u.searchParams.get('seasonId')||u.searchParams.get('season')||'').trim();
  var data=scan();
  if(!data||!data.picks||!data.picks.length){{return toast('No picks found yet — wait for a pick, then tap sync again.');}}
  data.picks.sort(function(a,b){{return a.overallPickNumber-b.overallPickNumber;}});
  fetch(O+'/api/draft/espn-relay',{{method:'POST',mode:'cors',credentials:'omit',headers:{{'Content-Type':'application/json','Authorization':'Bearer '+T}},body:JSON.stringify({{leagueId:leagueId,season:season,inProgress:data.inProgress!==false,drafted:!!data.drafted,picks:data.picks,source:{source_js}}})}})
  .then(function(r){{return r.json().then(function(b){{return{{ok:r.ok,b:b}};}});}})
  .then(function(res){{if(!res.ok){{toast((res.b&&res.b.error)||'Sync failed');return;}}toast('Synced '+(res.b.picks?res.b.picks.length:data.picks.length)+' picks to BR Fantasy');}})
  .catch(function(){{toast('Network error syncing to BR Fantasy');}});
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
