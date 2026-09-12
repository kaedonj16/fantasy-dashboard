// Shared accessible custom select enhancer. Keep this block byte-identical in app.js.
(function () {
  if (window.__brCustomSelectInstalled) return;
  window.__brCustomSelectInstalled = true;
  var seq = 0, openWrap = null, ARROW = '<svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true"><path d="M1 1L5 5L9 1" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>';
  function labelText(sel) {
    var labelled = sel.getAttribute('aria-labelledby');
    if (labelled) return labelled.split(/\s+/).map(function(id){ var e=document.getElementById(id); return e ? e.textContent.trim() : ''; }).filter(Boolean).join(' ');
    if (sel.getAttribute('aria-label')) return sel.getAttribute('aria-label');
    var label = sel.id && document.querySelector('label[for="' + CSS.escape(sel.id) + '"]');
    if (!label) label = sel.closest('label');
    return label ? label.textContent.trim() : (sel.name || 'Select option');
  }
  function initOne(sel) {
    if (sel._csdDone || sel.multiple || sel.hasAttribute('data-no-custom')) return;
    sel._csdDone = true;
    var cs = getComputedStyle(sel), wrap=document.createElement('div'), trigger=document.createElement('button'), list=document.createElement('div'), error=document.createElement('div');
    wrap.className='csd-wrap'; if(sel.style.display==='none') wrap.style.display='none';
    var mw=parseFloat(cs.minWidth); if(mw>0) wrap.style.minWidth=mw+'px';
    var listId='csd-list-'+(++seq); trigger.type='button'; trigger.className='csd-trigger';
    trigger.setAttribute('role','combobox'); trigger.setAttribute('aria-haspopup','listbox'); trigger.setAttribute('aria-expanded','false'); trigger.setAttribute('aria-controls',listId);
    trigger.style.fontWeight=cs.fontWeight; trigger.style.borderRadius=cs.borderRadius; trigger.style.paddingTop=cs.paddingTop; trigger.style.paddingBottom=cs.paddingBottom; trigger.style.paddingLeft=cs.paddingLeft;
    trigger.innerHTML='<span class="csd-value"></span><span class="csd-arrow">'+ARROW+'</span>';
    list.id=listId; list.className='csd-list'; list.setAttribute('role','listbox'); list.style.display='none';
    error.id=listId+'-error'; error.className='csd-error'; error.setAttribute('role','alert'); error.hidden=true;
    sel.parentNode.insertBefore(wrap,sel); wrap.append(trigger,list,error,sel);
    var valueEl=trigger.querySelector('.csd-value'), isOpen=false, focusIndex=-1, typeBuffer='', typeTimer;
    function disabled(opt){ return opt.disabled || (opt.parentElement && opt.parentElement.tagName==='OPTGROUP' && opt.parentElement.disabled); }
    function rebuild(){
      list.innerHTML=''; Array.from(sel.children).forEach(function(child){
        if(child.tagName==='OPTGROUP') { var gl=document.createElement('div'); gl.className='csd-group-label'; gl.textContent=child.label; gl.setAttribute('role','presentation'); list.appendChild(gl); Array.from(child.children).forEach(add); }
        else if(child.tagName==='OPTION') add(child);
      }); sync();
      function add(opt){ var el=document.createElement('div'), off=disabled(opt), selected=opt.selected; el.id=listId+'-opt-'+Array.from(sel.options).indexOf(opt); el.className='csd-option'+(off?' is-disabled':'')+(selected?' is-selected':''); el.dataset.value=opt.value; el.textContent=opt.textContent.trim(); el.setAttribute('role','option'); el.setAttribute('aria-selected',selected?'true':'false'); if(off) el.setAttribute('aria-disabled','true'); list.appendChild(el); }
    }
    function sync(){
      var opt=sel.options[sel.selectedIndex]; valueEl.textContent=opt?opt.textContent.trim():'';
      trigger.disabled=!!sel.disabled; trigger.setAttribute('aria-disabled',sel.disabled?'true':'false');
      if(sel.required) trigger.setAttribute('aria-required','true'); else trigger.removeAttribute('aria-required');
      var bad=!sel.disabled && !sel.checkValidity(); trigger.setAttribute('aria-invalid',bad?'true':'false'); wrap.classList.toggle('is-invalid',bad);
      if(!bad){error.hidden=true;error.textContent='';trigger.removeAttribute('aria-describedby');}
      var label=labelText(sel); trigger.setAttribute('aria-label',label+(valueEl.textContent?': '+valueEl.textContent:''));
      Array.from(list.querySelectorAll('[role=option]')).forEach(function(el){ var on=el.dataset.value===sel.value; el.classList.toggle('is-selected',on); el.setAttribute('aria-selected',on?'true':'false'); });
      if(sel.disabled) close();
    }
    function enabled(){ return Array.from(list.querySelectorAll('.csd-option:not(.is-disabled)')); }
    function focusAt(i){ var all=enabled(); if(!all.length)return; focusIndex=Math.max(0,Math.min(i,all.length-1)); all.forEach(function(e,n){e.classList.toggle('is-focused',n===focusIndex);}); trigger.setAttribute('aria-activedescendant',all[focusIndex].id); all[focusIndex].scrollIntoView({block:'nearest'}); }
    function position(){ var r=trigger.getBoundingClientRect(), vv=window.visualViewport, vh=vv?vv.height:innerHeight, vw=vv?vv.width:innerWidth, below=vh-r.bottom-8, above=r.top-8, up=above>below&&below<120; list.style.minWidth=r.width+'px'; list.style.maxWidth=Math.max(160,vw-16)+'px'; list.style.left=Math.max(8,Math.min(r.left,vw-(list.offsetWidth||r.width)-8))+'px'; list.style.top=up?'auto':(r.bottom+4)+'px'; list.style.bottom=up?(vh-r.top+4)+'px':'auto'; list.style.maxHeight=Math.max(80,Math.min(280,up?above:below))+'px'; }
    function open(){ if(isOpen||sel.disabled)return; if(openWrap&&openWrap!==wrap)openWrap._csdClose(); openWrap=wrap; isOpen=true; wrap.classList.add('is-open'); trigger.setAttribute('aria-expanded','true'); list.style.display='block'; position(); requestAnimationFrame(function(){list.classList.add('is-visible');}); var all=enabled(), ix=all.findIndex(function(e){return e.dataset.value===sel.value;}); focusAt(ix<0?0:ix); }
    function close(){ if(!isOpen)return; isOpen=false; if(openWrap===wrap)openWrap=null; wrap.classList.remove('is-open'); trigger.setAttribute('aria-expanded','false'); trigger.removeAttribute('aria-activedescendant'); list.classList.remove('is-visible'); setTimeout(function(){if(!isOpen)list.style.display='none';},150); }
    function pick(el){ if(!el||el.classList.contains('is-disabled')||sel.disabled)return; sel.value=el.dataset.value; sel.dispatchEvent(new Event('change',{bubbles:true})); sync(); close(); trigger.focus(); }
    trigger.addEventListener('click',function(e){e.stopPropagation();isOpen?close():open();});
    list.addEventListener('click',function(e){pick(e.target.closest('.csd-option'));});
    trigger.addEventListener('keydown',function(e){ var all=enabled(), k=e.key;
      if(k==='ArrowDown'||k==='ArrowUp'){e.preventDefault();if(!isOpen)open();else focusAt(focusIndex+(k==='ArrowDown'?1:-1));}
      else if(k==='Home'||k==='End'){e.preventDefault();if(!isOpen)open();focusAt(k==='Home'?0:all.length-1);}
      else if(k==='Enter'||k===' '){e.preventDefault();if(isOpen)pick(all[focusIndex]);else open();}
      else if(k==='Escape'){e.preventDefault();close();trigger.focus();} else if(k==='Tab')close();
      else if(k.length===1&&!e.ctrlKey&&!e.metaKey&&!e.altKey){ typeBuffer+=k.toLowerCase(); clearTimeout(typeTimer); typeTimer=setTimeout(function(){typeBuffer='';},600); var ix=all.findIndex(function(x){return x.textContent.trim().toLowerCase().startsWith(typeBuffer);}); if(ix>=0){e.preventDefault();if(!isOpen)open();focusAt(ix);} }
    });
    // Preserve native label activation after the select is visually replaced.
    var explicit=sel.id&&document.querySelector('label[for="'+CSS.escape(sel.id)+'"]'); if(explicit) explicit.addEventListener('click',function(e){if(e.target!==trigger){e.preventDefault();trigger.focus();}});
    sel.addEventListener('invalid',function(e){
      e.preventDefault(); sync(); error.textContent=sel.validationMessage||'Choose an option.'; error.hidden=false; trigger.setAttribute('aria-describedby',error.id); trigger.setAttribute('aria-invalid','true'); wrap.classList.add('is-invalid'); trigger.focus();
    });
    sel.addEventListener('change',sync); new MutationObserver(rebuild).observe(sel,{childList:true,subtree:true,attributes:true,attributeFilter:['disabled','selected','label']}); new MutationObserver(function(){wrap.style.display=sel.style.display==='none'?'none':'';sync();}).observe(sel,{attributes:true,attributeFilter:['style','disabled','required','aria-label','aria-labelledby']});
    wrap._csdClose=close; wrap._csdReposition=function(){if(isOpen)position();}; rebuild();
  }
  document.addEventListener('click',function(){if(openWrap)openWrap._csdClose();});
  addEventListener('scroll',function(){if(openWrap)openWrap._csdReposition();},true); addEventListener('resize',function(){if(openWrap)openWrap._csdReposition();}); if(window.visualViewport) visualViewport.addEventListener('resize',function(){if(openWrap)openWrap._csdReposition();});
  window.initCustomSelects=function(root){(root||document).querySelectorAll('select:not([data-no-custom]):not([multiple])').forEach(initOne);};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',function(){window.initCustomSelects();});else window.initCustomSelects();
}());
