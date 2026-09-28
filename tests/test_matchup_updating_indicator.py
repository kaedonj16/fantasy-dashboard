"""Regression test: the "Updating matchup…" indicator must not get stuck.

Drives the real renderMatchup/stateNote/_lsLoad code from static/app.js (the
portfolio-cards IIFE) in Node with a minimal fake DOM:

1. A pending poll after populated data shows "Updating matchup…".
2. If the backend stays cold, the note is dropped after ~90s instead of
   sitting next to populated scores forever.
3. A successful render clears the note.
4. A failure replaces the note text (and a later pending restarts the clock).
5. The League Scores "Retry" button actually refetches (it was dead).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MARKER = "/* Portfolio cards: one generation owns requests, retries, polling and listeners. */"

_NODE_PRELUDE = r"""
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function dataKey(attr){return attr.slice(5).replace(/-([a-z])/g,function(_,c){return c.toUpperCase();});}
function matches(el,sel){
  var m=/^\[([^\]="]+)(?:="([^"]*)")?\]$/.exec(sel);
  if(!m) return false;
  var v=el._attrs[m[1]];
  if(v===undefined) return false;
  return m[2]===undefined||v===m[2];
}
function collect(root,sel,out){
  root.children.forEach(function(c){if(matches(c,sel))out.push(c);collect(c,sel,out);});
}
function makeEl(tag){
  var el={
    tagName:String(tag||'div').toUpperCase(),
    children:[],parent:null,dataset:{},_attrs:{},
    _html:null,_text:'',className:'',hidden:false,style:{},
    _listeners:{},isConnected:true,
    setAttribute:function(k,v){k=String(k);v=String(v);this._attrs[k]=v;if(k.indexOf('data-')===0)this.dataset[dataKey(k)]=v;},
    getAttribute:function(k){return this._attrs[String(k)];},
    removeAttribute:function(k){k=String(k);delete this._attrs[k];if(k.indexOf('data-')===0)delete this.dataset[dataKey(k)];},
    appendChild:function(c){c.parent=this;this.children.push(c);return c;},
    addEventListener:function(t,fn){(this._listeners[t]=this._listeners[t]||[]).push(fn);},
    querySelector:function(sel){var out=[];collect(this,sel,out);return out[0]||null;},
    querySelectorAll:function(sel){var out=[];collect(this,sel,out);return out;},
    closest:function(sel){var n=this;while(n){if(matches(n,sel))return n;n=n.parent;}return null;},
    contains:function(n){while(n){if(n===this)return true;n=n.parent;}return false;},
    remove:function(){if(this.parent){var i=this.parent.children.indexOf(this);if(i>=0)this.parent.children.splice(i,1);this.parent=null;}}
  };
  Object.defineProperty(el,'innerHTML',{
    get:function(){return this._html!==null?this._html:esc(this._text);},
    set:function(v){var self=this;this._html=String(v);this.children.forEach(function(c){c.parent=null;});this.children=[];}
  });
  Object.defineProperty(el,'textContent',{
    get:function(){return this._text;},
    set:function(v){this._text=String(v);this._html=null;}
  });
  return el;
}
var document={createElement:makeEl,addEventListener:function(){}};
var window={};
"""

_NODE_DRIVER = r"""
var realNow=Date.now, now=1000000;
Date.now=function(){return now;};
var T=window.__brPortfolioCardsTest;
function assert(c,msg){if(!c){console.error('FAIL: '+msg);process.exit(1);}console.log('ok: '+msg);}

var SUCCESS={live:true,week:3,status:'final',you:{score:179.6,proj:170},opp:{name:'Pittsburgh Pilots',score:144.2,proj:150},result:'W',margin:35.4};

// 1. pending after populated data -> "Updating matchup…" note appears
var slot=makeEl('div');
slot.dataset.matchupGood='true';
slot.innerHTML='<div class="populated">scores</div>';
var r=T.renderMatchup(slot,{pending:true});
assert(r===false,'pending render returns false');
var note=slot.querySelector('[data-matchup-state]');
assert(note&&note.textContent==='Updating matchup…','updating note shown on pending');
assert(slot.children.indexOf(note)>=0,'note appended without wiping populated scores');

// 2. still pending just under the bound -> note stays
now+=89000;
T.renderMatchup(slot,{pending:true});
assert(slot.querySelector('[data-matchup-state]')!==null,'note persists before the 90s bound');

// 3. still pending past the bound -> note dropped, scores stand alone
now+=2000;
T.renderMatchup(slot,{pending:true});
assert(slot.querySelector('[data-matchup-state]')===null,'note dropped after ~90s of pending');

// 4. success after pending -> note gone, scores rendered
slot=makeEl('div');
slot.dataset.matchupGood='true';
T.renderMatchup(slot,{pending:true});
assert(slot.querySelector('[data-matchup-state]')!==null,'note shown before success');
r=T.renderMatchup(slot,SUCCESS);
assert(r===true,'success render returns true');
assert(slot.querySelector('[data-matchup-state]')===null,'note cleared on success');
assert(slot.innerHTML.indexOf('179.60')!==-1,'populated scores rendered');

// 5. failure swaps the note text; a later pending restarts the clock
slot=makeEl('div');
slot.dataset.matchupGood='true';
slot.innerHTML='<div>scores</div>';
T.renderMatchup(slot,{pending:true});
now+=1000000;
r=T.renderMatchup(slot,{failed:true,message:'boom'});
note=slot.querySelector('[data-matchup-state]');
assert(note&&note.textContent==='boom','failed note replaces the updating note');
T.renderMatchup(slot,{pending:true});
assert(slot.querySelector('[data-matchup-state]')!==null,'pending note reappears after failure with a fresh timestamp');

// 6. League Scores retry button refetches (it was previously dead)
var fetchCalls=0, fetchImpl=function(){fetchCalls++;return Promise.reject(new Error('nope'));};
global.fetch=function(){return fetchImpl.apply(null,arguments);};
slot=makeEl('div');
slot.dataset.matchupGood='true';
T.renderMatchup(slot,SUCCESS); // wires the delegated click listener
var view=makeEl('div');view.setAttribute('data-ls-view','league');slot.appendChild(view);
var card=makeEl('div');card.setAttribute('data-summary-card','');
card.dataset.platform='sleeper';card.dataset.leagueId='123';card.dataset.season='2026';
slot.parent=card;
var btn=makeEl('button');btn.setAttribute('data-ls-retry','');view.appendChild(btn);
function click(t){(slot._listeners.click||[]).forEach(function(fn){fn({target:t,preventDefault:function(){}});});}
click(btn);
setTimeout(function(){
  assert(fetchCalls===1,'retry button triggers a league-scores refetch');
  assert(view.innerHTML.indexOf('data-ls-retry')!==-1,'failed load shows a retry button');
  // the failed load's innerHTML replaced the button node; a real DOM would
  // have parsed a fresh one, so do the same here
  var btn2=makeEl('button');btn2.setAttribute('data-ls-retry','');view.appendChild(btn2);
  fetchImpl=function(){fetchCalls++;return Promise.resolve({json:function(){return Promise.resolve({week:3,matchups:[{left:{name:'A',score:100,proj:100},right:{name:'B',score:90,proj:90},status:'final',is_you:true}]});}});};
  click(btn2);
  setTimeout(function(){
    assert(fetchCalls===2,'second retry refetches again');
    assert(view.innerHTML.indexOf('ls-list')!==-1,'league scores render after retry');
    console.log('ALL PASS');
  },50);
},50);
"""


def _extract_iife() -> str:
    path = os.path.join(_REPO_ROOT, "static", "app.js")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    idx = src.find(_MARKER)
    assert idx != -1, "portfolio-cards IIFE marker not found in static/app.js"
    return src[idx:]


def test_matchup_updating_indicator_not_stuck():
    if not shutil.which("node"):
        pytest.skip("node not available")
    iife = _extract_iife()
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(_NODE_PRELUDE + "\n" + iife + "\n" + _NODE_DRIVER)
        tmp = fh.name
    try:
        proc = subprocess.run(["node", tmp], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(tmp)
    assert proc.returncode == 0, "node driver failed:\n" + proc.stdout + "\n" + proc.stderr
    assert "ALL PASS" in proc.stdout, "driver did not complete:\n" + proc.stdout
