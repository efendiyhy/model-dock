/* ModelDock 0.4 | EK Instrumentation. Frontend only; retains the existing FastAPI backend. */
'use strict';
const icons={
 book:'<path d="M12 6c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1ZM12 6v15"/>',
 logout:'<path d="M9 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h4M16 17l5-5-5-5m5 5H9"/>',
 upload:'<path d="M12 16V3m-5 5 5-5 5 5M4 17v3h16v-3"/>',
 brain:'<path d="M12 20V5m0 3c-4-6-11 0-8 5-3 5 4 10 8 6m0-11c4-6 11 0 8 5-3 5 4 10 8 6"/>',
 chart:'<path d="m3 17 6-6 4 4 8-9"/>',
 copy:'<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
 play:'<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4V8Z"/>',
 key:'<circle cx="16" cy="8" r="5"/><path d="m12 12-8 8v-4h4v-4"/>',
 arrow:'<path d="M5 12h14m-6-6 6 6-6 6"/>',
 refresh:'<path d="M3 11a9 9 0 0 1 15-6l2 2M20 3v4h-4M21 13a9 9 0 0 1-15 6l-2-2M4 21v-4h4"/>',
 chevron:'<path d="m6 9 6 6 6-6"/>',
 terminal:'<path d="m5 7 5 5-5 5m7 0h7"/>',
 circle:'<circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8v.2"/>',
 search:'<circle cx="11" cy="11" r="7"/><path d="m16 16 5 5"/>',
 layers:'<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 10h8M8 14h8"/>',
 activity:'<path d="M3 12h4l3-7 4 14 3-7h4"/>',
 power:'<path d="M12 3v9m-5-6a9 9 0 1 0 10 0"/>',
 code:'<path d="m8 8-4 4 4 4m8-8 4 4-4 4m-3-11-2 14"/>'
};
const ico=(name,size=18)=>`<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]||icons.brain}</svg>`;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $=s=>document.querySelector(s);
const state={csrf:'',username:'',models:[],overview:{models:0,running:0,requests:0},key:null,logs:[],showKey:false,detailSlug:null,query:'',filter:'all'};
let toastTimer;
function toast(message,error=false){const el=$('#toast');el.textContent=message;el.className='toast show'+(error?' error':'');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.className='toast',3500)}
function showLogin(){state.csrf='';state.key=null;$('#app-shell').classList.add('is-hidden');$('#login-screen').classList.remove('is-hidden');$('#login-password').value='';$('#login-error').hidden=true;for(const id of ['#upload-dialog','#model-dialog','#confirm-dialog']){$(id).close?.()}}
function showApp(username,csrf){state.username=username;state.csrf=csrf;$('#login-screen').classList.add('is-hidden');$('#app-shell').classList.remove('is-hidden')}
async function api(url,opts={}){
 const method=(opts.method||'GET').toUpperCase();
 const headers=new Headers(opts.headers||{});
 if(['POST','PATCH','PUT','DELETE'].includes(method)&&url!=='/api/auth/login'&&state.csrf)headers.set('X-CSRF-Token',state.csrf);
 const response=await fetch(url,{...opts,headers,credentials:'same-origin',cache:'no-store'});
 let body={};try{body=await response.json()}catch{body={detail:'Server mengirim respons tidak valid.'}}
 if(!response.ok){if(response.status===401&&!['/api/auth/login','/api/auth/me'].includes(url))showLogin();let message=body.detail||body.message||`HTTP ${response.status}`;if(Array.isArray(message))message=message.map(x=>x.msg).join(', ');throw new Error(message)}
 return body;
}
async function loadData(){const [overview,models,key,logs]=await Promise.all([api('/api/overview'),api('/api/models'),api('/api/key'),api('/api/requests?limit=6')]);Object.assign(state,{overview,models,key,logs})}
const urlFor=m=>`${location.origin}/v1/${m.slug}/predict`;
const num=x=>(Number(x)||0).toLocaleString('id-ID');
const dt=x=>new Date(x).toLocaleString('id-ID',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});
function modelRow(m){return `<article class="model-row" data-model-slug="${esc(m.slug)}">
 <div class="model-identity"><div class="model-symbol ${m.task==='regression'?'alt':''}">${ico(m.task==='regression'?'chart':'brain')}</div><div class="model-info"><strong title="${esc(m.name)}">${esc(m.name)}</strong><small>${m.task==='classification'?'Klasifikasi':'Regresi'} · ${m.features.length} fitur</small></div></div>
 <div class="endpoint-block" title="/v1/${esc(m.slug)}/predict"><span class="method-tag">POST</span><code class="endpoint-path">/v1/${esc(m.slug)}/predict</code></div>
 <div class="status ${m.active?'':'off'}">${m.active?'Aktif':'Nonaktif'}</div>
 <div class="request-count">${num(m.requests)}</div>
 <div class="model-actions"><button type="button" class="button button-outline button-small button-icon" data-action="copy-url" data-slug="${esc(m.slug)}" title="Salin endpoint ${esc(m.name)}" aria-label="Salin endpoint ${esc(m.name)}">${ico('copy',16)}</button><button type="button" class="button button-outline button-small button-icon" data-action="toggle-row" data-slug="${esc(m.slug)}" title="${m.active?'Nonaktifkan':'Aktifkan'} ${esc(m.name)}" aria-label="${m.active?'Nonaktifkan':'Aktifkan'} ${esc(m.name)}">${ico('power',16)}</button><button type="button" class="button button-subtle button-small" data-action="detail" data-slug="${esc(m.slug)}">Kelola ${ico('arrow',14)}</button></div>
 </article>`}
function filteredModels(){const query=state.query.trim().toLowerCase();return state.models.filter(m=>(state.filter==='all'||(state.filter==='active'?m.active:!m.active))&&(!query||[m.name,m.slug,m.task,...m.features].some(t=>String(t).toLowerCase().includes(query))))}
function listHtml(){const selected=filteredModels();if(!state.models.length)return `<div class="empty"><div class="model-symbol">${ico('layers')}</div><h3>Belum ada model tersimpan</h3><p>Unggah model yang sudah dilatih untuk membuat endpoint FastAPI pertamamu.</p><button class="button button-primary" type="button" data-action="upload">${ico('upload')} Upload model</button></div>`;
 if(!selected.length)return `<div class="empty-filter"><strong>Model tidak ditemukan</strong><p>Coba kata pencarian atau filter yang lain.</p><button type="button" class="inline-link" data-action="clear-filters">Reset pencarian</button></div>`;
 return selected.map(modelRow).join('')}
function updateModelManager(){const list=$('#model-rows');if(!list)return;list.innerHTML=listHtml();const count=$('#visible-model-count');if(count)count.textContent=`Menampilkan ${filteredModels().length} dari ${state.models.length} model tersimpan`;document.querySelectorAll('[data-filter]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.filter===state.filter)))}
function helpBlock(){return `<details class="fold" id="guide-details"><summary>${ico('book')} <span class="fold-title">Cara menggunakan<small>Upload, uji prediksi, dan gunakan API</small></span> ${ico('chevron',16).replace('<svg','<svg class="chevron"')}</summary><div class="fold-body">
 <div class="guide-body">
  <div class="guide-step"><span class="step-num">01</span><h3>Upload model</h3><p>Pilih file model .joblib atau .pkl dari Python yang sudah selesai dilatih.</p></div>
  <div class="guide-step"><span class="step-num">02</span><h3>Uji prediksi</h3><p>Klik <b>Uji & Kelola</b> pada model, masukkan JSON, dan jalankan prediksi.</p></div>
  <div class="guide-step"><span class="step-num">03</span><h3>Gunakan API</h3><p>Salin URL endpoint dan API key. Kirim POST JSON dari backend aplikasi lain.</p></div>
 </div>
 <div class="guide-example">Contoh request (model Iris):<pre class="code-sample">POST /v1/iris-demo/predict
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "inputs": {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }
}</pre><a class="inline-link" href="/docs" target="_blank" rel="noopener">Buka dokumentasi Swagger ↗</a></div>
 <div class="notice"><strong>Keamanan prototipe</strong>File pickle/joblib bisa mengeksekusi kode saat dimuat. Hanya upload file tepercaya. Jangan membuka versi ini ke internet publik sebelum isolasi model dan pengamanan server. API key disimpan di backend, bukan frontend publik.</div>
 </div></details>`}
function keyBlock(){const key=state.key?.api_key||'';return `<details class="fold" id="key-details"><summary>${ico('key')} <span class="fold-title">Akses API<small>Kelola API key dan integrasi dengan aplikasi lain</small></span> ${ico('chevron',16).replace('<svg','<svg class="chevron"')}</summary><div class="fold-body"><p>Gunakan header <code>X-API-Key</code> ketika website lain memanggil endpoint prediksi.</p>
 <div class="key-line"><code>${state.showKey?esc(key):'md_live_••••••••••••••••••••••••'}</code><button type="button" class="button button-outline button-small" data-action="reveal">${state.showKey?'Sembunyikan':'Lihat'}</button><button type="button" class="button button-subtle button-small" data-action="copy-key">${ico('copy',14)} Salin key</button></div>
 <div class="panel-note">Jaga kerahasiaan key. <button class="inline-link" type="button" data-action="rotate-key">Buat API key baru</button> — key lama akan otomatis tidak berlaku.</div></div></details>`}
function activityBlock(){return `<details class="fold" id="logs-details"><summary>${ico('terminal')} Aktivitas API terakhir ${ico('chevron',16).replace('<svg','<svg class="chevron"')}</summary><div class="fold-body"><p>Riwayat request prediksi terbaru. <button type="button" class="inline-link" data-action="refresh">Refresh</button></p>
 ${state.logs.length?state.logs.map(x=>`<div class="log-row"><div><strong>${esc(x.model_slug)}</strong><small>${dt(x.at)} · ${x.source==='api'?'API eksternal':'Tes dashboard'}</small></div><span class="log-status ${x.status>=400?'err':''}">${x.status} · ${Number(x.duration_ms).toFixed(1)}ms</span></div>`).join(''):'<p class="muted">Belum ada request. Coba model demo Iris.</p>'}
 </div></details>`}
function dashboard(){const o=state.overview;return `
 <section class="workspace-intro" aria-labelledby="dashboard-title"><div><span class="eyebrow">MODELDOCK WORKSPACE</span><h1 class="workspace-title" id="dashboard-title">Model manager</h1><p class="workspace-sub">Semua model yang sudah dilatih dan diunggah, siap digunakan melalui FastAPI.</p></div><button class="button button-primary workspace-cta" data-action="upload" type="button">${ico('upload')} Upload model baru</button></section>
 <section class="metrics" aria-label="Ringkasan platform"><div class="metric-card"><div class="metric-top"><span>Total model</span>${ico('layers')}</div><div class="metric-number">${num(o.models)}</div><div class="metric-hint">Tersimpan di server</div></div><div class="metric-card"><div class="metric-top"><span>Model aktif</span>${ico('activity')}</div><div class="metric-number">${num(o.running)}</div><div class="metric-hint">Endpoint tersedia</div></div><div class="metric-card"><div class="metric-top"><span>API requests</span>${ico('code')}</div><div class="metric-number">${num(o.requests)}</div><div class="metric-hint">Total panggilan tercatat</div></div></section>
 <section class="manager" aria-labelledby="models-heading"><header class="manager-head"><div><div class="manager-heading"><h2 id="models-heading">Model tersimpan</h2><span class="count-pill">${num(state.models.length)}</span></div><p class="manager-description">Cari, periksa, uji, aktifkan, dan kelola modelmu.</p></div><button type="button" class="button button-outline button-small" data-action="refresh" title="Perbarui data">${ico('refresh')} <span>Refresh</span></button></header>
 <div class="manager-toolbar"><label class="search-field"><span class="sr-only">Cari model</span>${ico('search',18)}<input id="model-search" class="input" type="search" autocomplete="off" placeholder="Cari nama, endpoint, atau fitur..." value="${esc(state.query)}" aria-label="Cari model tersimpan"></label>
 <div class="filter-group" role="group" aria-label="Filter status model"><button class="filter-button" type="button" data-filter="all" aria-pressed="${state.filter==='all'}">Semua</button><button class="filter-button" type="button" data-filter="active" aria-pressed="${state.filter==='active'}">Aktif</button><button class="filter-button" type="button" data-filter="inactive" aria-pressed="${state.filter==='inactive'}">Nonaktif</button></div></div>
 <div class="manager-labels" aria-hidden="true"><span>Nama model</span><span>Endpoint</span><span>Status</span><span>Requests</span><span>Aksi</span></div><div id="model-rows" class="manager-list">${listHtml()}</div><div class="manager-foot"><span id="visible-model-count">Menampilkan ${filteredModels().length} dari ${state.models.length} model tersimpan</span><button type="button" class="inline-link" data-action="help">Butuh bantuan? ${ico('arrow',14)}</button></div></section>
 <section class="utility-grid" aria-label="Akses API dan panduan">${keyBlock()}${helpBlock()}</section><section class="activity-stack" aria-label="Aktivitas terakhir">${activityBlock()}</section>`}
function render(){if(!state.csrf)return;const helpOpen=$('#guide-details')?.open||false,keyOpen=$('#key-details')?.open||false,logsOpen=$('#logs-details')?.open||false;$('#main-content').innerHTML=dashboard();if(helpOpen)$('#guide-details').open=true;if(keyOpen)$('#key-details').open=true;if(logsOpen)$('#logs-details').open=true}
function openHelp(){const el=$('#guide-details');if(!el)return;el.open=true;el.scrollIntoView({behavior:'smooth',block:'center'})}
function openKey(){const el=$('#key-details');if(!el)return;el.open=true;el.scrollIntoView({behavior:'smooth',block:'center'})}
function renderDetail(slug){const m=state.models.find(x=>x.slug===slug);if(!m){toast('Model tidak ditemukan',true);return;}state.detailSlug=slug;const sample=JSON.stringify({inputs:m.sample_input},null,2);$('#detail-content').innerHTML=`<div class="dialog-header"><div><div class="overline">MODEL DETAILS</div><h2 id="detail-title">${esc(m.name)}</h2><p>${m.task==='classification'?'Klasifikasi':'Regresi'} · ${m.features.length} fitur · ${m.active?'Aktif':'Nonaktif'}</p></div><button type="button" class="dialog-close" data-action="close-detail" aria-label="Tutup">×</button></div>
 <div class="detail-endpoint"><code>${esc(urlFor(m))}</code><button type="button" class="button button-outline button-small" data-action="copy-url" data-slug="${esc(m.slug)}">${ico('copy',14)} Salin URL</button></div>
 <div class="detail-section"><h3>Fitur input (${m.features.length})</h3><div class="feature-tags">${m.features.map(f=>`<span>${esc(f)}</span>`).join('')}</div></div>
 <div class="detail-section"><h3>Uji prediksi</h3><p class="detail-meta">Ubah nilai JSON di bawah lalu jalankan inferensi Python.</p><textarea class="json-input" id="predict-input" spellcheck="false">${esc(sample)}</textarea><div class="detail-actions" style="margin-top:11px"><button type="button" class="button button-primary button-small" data-action="run-test">${ico('play',15)} Jalankan prediksi</button><button type="button" class="button button-outline button-small" data-action="reset-input">Reset</button></div><h3 style="margin-top:20px">Hasil prediksi</h3><pre class="result-box" id="predict-output">Hasil akan muncul di sini.</pre></div>
 <div class="detail-footer"><button type="button" class="button button-outline button-small" data-action="toggle-model">${m.active?'Hentikan model':'Aktifkan model'}</button><button type="button" class="button button-danger button-small" data-action="delete-model">Hapus model</button><button type="button" class="button button-outline button-small" data-action="close-detail">Tutup</button></div>`;
 if(!$('#model-dialog').open)$('#model-dialog').showModal();}
async function login(event){event.preventDefault();$('#login-error').hidden=true;const button=$('#login-submit');button.disabled=true;button.textContent='Memverifikasi...';try{const auth=await api('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:$('#login-username').value.trim(),password:$('#login-password').value})});showApp(auth.username,auth.csrf_token);await loadData();render();toast('Login berhasil.')}catch(err){const p=$('#login-error');p.hidden=false;p.textContent=err.message;if(state.csrf)showLogin()}finally{button.disabled=false;button.innerHTML='Masuk <span aria-hidden="true">→</span>'}}
async function logout(){try{await api('/api/auth/logout',{method:'POST'})}catch{}showLogin();toast('Berhasil logout.')}
function showUpload(){const modal=$('#upload-dialog');if(!modal.open)modal.showModal();$('#model-file').focus()}
async function upload(event){event.preventDefault();const file=$('#model-file').files[0];if(!file){toast('Pilih file model terlebih dahulu.',true);return}const button=$('#upload-submit');button.disabled=true;button.textContent='Mengunggah...';try{const data=new FormData($('#upload-form'));const result=await api('/api/models',{method:'POST',body:data});$('#upload-dialog').close();$('#upload-form').reset();$('#file-name').textContent='Pilih model dari komputer';$('#file-subtitle').textContent='.joblib / .pkl / .pickle · maks. 35 MB';await loadData();render();renderDetail(result.slug);toast('Model berhasil diunggah. API sudah tersedia.')}catch(err){toast(err.message,true)}finally{button.disabled=false;button.innerHTML='Upload & Buat API <span>→</span>'}}
async function runTest(){const m=state.models.find(x=>x.slug===state.detailSlug);if(!m)return;const output=$('#predict-output');let body;try{body=JSON.parse($('#predict-input').value)}catch(err){toast('Format JSON tidak valid: '+err.message,true);return}if(!body||!Object.prototype.hasOwnProperty.call(body,'inputs')){toast('JSON harus mempunyai properti "inputs".',true);return}output.textContent='Menjalankan prediksi...';try{const result=await api(`/api/models/${encodeURIComponent(m.slug)}/test`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});output.textContent=JSON.stringify(result,null,2);await loadData();render();toast('Prediksi berhasil.')}catch(err){output.textContent=JSON.stringify({error:err.message},null,2);toast(err.message,true)}}
async function toggleModel(){const m=state.models.find(x=>x.slug===state.detailSlug);if(!m)return;const activating=!m.active;try{await api(`/api/models/${encodeURIComponent(m.slug)}/status`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({active:activating})});await loadData();render();if($('#model-dialog').open)renderDetail(m.slug);toast(activating?'Model diaktifkan.':'Model dihentikan.')}catch(err){toast(err.message,true)}}
async function deleteModel(){const m=state.models.find(x=>x.slug===state.detailSlug);if(!m)return;if(!await confirmChoice('Hapus model?',`Model \"${m.name}\" dan file tersimpannya akan dihapus permanen. Tindakan ini tidak dapat dibatalkan.`, 'Hapus model'))return;try{await api(`/api/models/${encodeURIComponent(m.slug)}`,{method:'DELETE'});$('#model-dialog').close();state.detailSlug=null;await loadData();render();toast('Model telah dihapus.')}catch(err){toast(err.message,true)}}
async function rotateKey(){if(!await confirmChoice('Ganti API key?', 'Key lama akan dinonaktifkan. Aplikasi lain harus menggunakan key baru untuk bisa mengakses endpoint.', 'Ganti key'))return;try{state.key=await api('/api/key/regenerate',{method:'POST'});state.showKey=true;render();$('#key-details').open=true;toast('API key baru dibuat.')}catch(err){toast(err.message,true)}}
let pendingConfirm=null;
function confirmChoice(title,message,acceptText){return new Promise(resolve=>{const el=$('#confirm-dialog');if(el.open)el.close();$('#confirm-title').textContent=title;$('#confirm-message').textContent=message;$('#confirm-accept').textContent=acceptText;pendingConfirm=resolve;el.showModal();})}
function finishConfirm(approved){const el=$('#confirm-dialog');if(el.open)el.close();if(pendingConfirm){const done=pendingConfirm;pendingConfirm=null;done(approved)}}
$('#confirm-dialog').addEventListener('close',()=>{if(pendingConfirm){const done=pendingConfirm;pendingConfirm=null;done(false)}});
async function clipboard(value){try{await navigator.clipboard.writeText(value)}catch{const t=document.createElement('textarea');t.value=value;document.body.append(t);t.select();document.execCommand('copy');t.remove()}toast('Disalin ke clipboard.')}
async function refresh(){const open=['guide-details','key-details','logs-details'].filter(id=>$('#'+id)?.open);await loadData();render();open.forEach(id=>$('#'+id).open=true);toast('Data diperbarui.')}
document.addEventListener('click',async event=>{const button=event.target.closest('[data-action]');if(!button)return;const action=button.dataset.action,slug=button.dataset.slug;
 if(action==='upload')showUpload();
 else if(action==='open-key')openKey();
 else if(action==='clear-filters'){state.query='';state.filter='all';const s=$('#model-search');if(s)s.value='';updateModelManager()}
 else if(action==='toggle-row'){state.detailSlug=slug;await toggleModel()}
 else if(action==='cancel-confirm')finishConfirm(false);
 else if(action==='accept-confirm')finishConfirm(true);
 else if(action==='help')openHelp();
 else if(action==='close-upload')$('#upload-dialog').close();
 else if(action==='close-detail')$('#model-dialog').close();
 else if(action==='detail')renderDetail(slug);
 else if(action==='copy-url'){const m=state.models.find(x=>x.slug===slug);if(m)await clipboard(urlFor(m))}
 else if(action==='copy-key')await clipboard(state.key?.api_key||'');
 else if(action==='reveal'){state.showKey=!state.showKey;render();$('#key-details').open=true}
 else if(action==='rotate-key')await rotateKey();
 else if(action==='run-test')await runTest();
 else if(action==='reset-input'){const m=state.models.find(x=>x.slug===state.detailSlug);if(m)$('#predict-input').value=JSON.stringify({inputs:m.sample_input},null,2)}
 else if(action==='toggle-model')await toggleModel();
 else if(action==='delete-model')await deleteModel();
 else if(action==='refresh')await refresh();
});
document.addEventListener('click',event=>{const filter=event.target.closest('[data-filter]');if(filter){state.filter=filter.dataset.filter;updateModelManager()}});
document.addEventListener('input',event=>{if(event.target.id==='model-search'){state.query=event.target.value;updateModelManager()}});
$('#login-form').addEventListener('submit',login);
$('#upload-form').addEventListener('submit',upload);
$('#logout-side').addEventListener('click',logout);
$('#brand-home').addEventListener('click',e=>{e.preventDefault();scrollTo({top:0,behavior:'smooth'})});
$('#toggle-password').addEventListener('click',()=>{const el=$('#login-password');el.type=el.type==='text'?'password':'text';$('#toggle-password').textContent=el.type==='password'?'Lihat':'Sembunyi'});
$('#model-file').addEventListener('change',()=>{const file=$('#model-file').files[0];if(!file)return;$('#file-name').textContent=file.name;$('#file-subtitle').textContent=`${(file.size/1024).toFixed(1)} KB · siap diunggah`;if(!$('#model-name').value)$('#model-name').value=file.name.replace(/\.(joblib|pkl|pickle)$/i,'').replace(/[_-]+/g,' ')});
document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=ico(el.dataset.icon));
(async function init(){try{const me=await api('/api/auth/me');showApp(me.username,me.csrf_token);await loadData();render()}catch{showLogin()}})();
