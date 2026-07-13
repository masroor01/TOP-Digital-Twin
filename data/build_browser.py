import json, os

with open('data/market_coverage_list.json', encoding='utf-8') as f:
    raw = f.read()

TEMPLATE_TOP = r"""<title>TOP Market Coverage</title>
<style>
  :root {
    --bg:#F4F6F1;--surface:#FFFFFF;--surface2:#EEF1EA;--border:#D4DDD0;
    --text:#1A2518;--text-muted:#5E7A5A;--accent:#1E5C37;--accent-light:#E4F0E8;
    --tag-full-bg:#D1EDDF;--tag-full-tx:#14582E;
    --tag-good-bg:#DCF0E7;--tag-good-tx:#1D6B3C;
    --tag-mod-bg:#FEF3C7;--tag-mod-tx:#7A5400;
    --tag-sht-bg:#FFE4CC;--tag-sht-tx:#863200;
    --tag-thin-bg:#FFD9D4;--tag-thin-tx:#8C1A14;
    --tomato:#C0392B;--onion:#7B2C8E;--potato:#A07020;
    --shadow:0 1px 4px rgba(0,0,0,.08);
  }
  @media(prefers-color-scheme:dark){:root{
    --bg:#0E1510;--surface:#182018;--surface2:#1F2B1E;--border:#2E3E2C;
    --text:#E2EAE0;--text-muted:#7A9A76;--accent:#4CAF77;--accent-light:#1A3326;
    --tag-full-bg:#123C22;--tag-full-tx:#78D8A0;
    --tag-good-bg:#1A4228;--tag-good-tx:#8FDAAA;
    --tag-mod-bg:#3D2E00;--tag-mod-tx:#F5CC5A;
    --tag-sht-bg:#3E1E00;--tag-sht-tx:#F5A060;
    --tag-thin-bg:#3E0E0A;--tag-thin-tx:#F59090;
    --tomato:#F08080;--onion:#C080E0;--potato:#D4A050;
    --shadow:0 1px 6px rgba(0,0,0,.35);
  }}
  :root[data-theme="light"]{
    --bg:#F4F6F1;--surface:#FFFFFF;--surface2:#EEF1EA;--border:#D4DDD0;
    --text:#1A2518;--text-muted:#5E7A5A;--accent:#1E5C37;--accent-light:#E4F0E8;
    --tag-full-bg:#D1EDDF;--tag-full-tx:#14582E;
    --tag-good-bg:#DCF0E7;--tag-good-tx:#1D6B3C;
    --tag-mod-bg:#FEF3C7;--tag-mod-tx:#7A5400;
    --tag-sht-bg:#FFE4CC;--tag-sht-tx:#863200;
    --tag-thin-bg:#FFD9D4;--tag-thin-tx:#8C1A14;
    --tomato:#C0392B;--onion:#7B2C8E;--potato:#A07020;
    --shadow:0 1px 4px rgba(0,0,0,.08);
  }
  :root[data-theme="dark"]{
    --bg:#0E1510;--surface:#182018;--surface2:#1F2B1E;--border:#2E3E2C;
    --text:#E2EAE0;--text-muted:#7A9A76;--accent:#4CAF77;--accent-light:#1A3326;
    --tag-full-bg:#123C22;--tag-full-tx:#78D8A0;
    --tag-good-bg:#1A4228;--tag-good-tx:#8FDAAA;
    --tag-mod-bg:#3D2E00;--tag-mod-tx:#F5CC5A;
    --tag-sht-bg:#3E1E00;--tag-sht-tx:#F5A060;
    --tag-thin-bg:#3E0E0A;--tag-thin-tx:#F59090;
    --tomato:#F08080;--onion:#C080E0;--potato:#D4A050;
    --shadow:0 1px 6px rgba(0,0,0,.35);
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);font-size:13px;line-height:1.5;min-height:100vh}
  .header{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 20px 10px;position:sticky;top:0;z-index:100;box-shadow:var(--shadow)}
  .header-top{display:flex;align-items:baseline;gap:14px;margin-bottom:10px;flex-wrap:wrap}
  .header-title{font-family:Georgia,serif;font-size:17px;font-weight:normal;color:var(--text);letter-spacing:-.01em}
  .header-subtitle{font-size:11px;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase}
  .summary-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
  .crop-chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer;border:2px solid transparent;transition:all .15s;user-select:none;background:var(--surface2);color:var(--text)}
  .crop-chip[data-crop="tomato"]{background:#FDE8E6;color:var(--tomato)}
  .crop-chip[data-crop="onion"]{background:#F2E5F7;color:var(--onion)}
  .crop-chip[data-crop="potato"]{background:#FEF2D4;color:var(--potato)}
  :root[data-theme="dark"] .crop-chip[data-crop="tomato"],:root[data-theme="dark"] .crop-chip[data-crop="tomato"]{background:#2E1008}
  :root[data-theme="dark"] .crop-chip[data-crop="onion"]{background:#200830}
  :root[data-theme="dark"] .crop-chip[data-crop="potato"]{background:#281A00}
  @media(prefers-color-scheme:dark){
    .crop-chip[data-crop="tomato"]{background:#2E1008}
    .crop-chip[data-crop="onion"]{background:#200830}
    .crop-chip[data-crop="potato"]{background:#281A00}
  }
  .crop-chip.active{border-color:currentColor}
  .crop-chip .count{font-weight:400;opacity:.75}
  .filter-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .search-wrap{position:relative;flex:1;min-width:160px;max-width:300px}
  .search-wrap svg{position:absolute;left:8px;top:50%;transform:translateY(-50%);opacity:.4;pointer-events:none}
  #searchInput{width:100%;padding:5px 8px 5px 28px;border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--text);font-size:12px;outline:none}
  #searchInput:focus{border-color:var(--accent)}
  select{padding:5px 9px;border:1px solid var(--border);border-radius:5px;background:var(--bg);color:var(--text);font-size:12px;outline:none;cursor:pointer}
  select:focus{border-color:var(--accent)}
  .cov-filters{display:flex;gap:4px;flex-wrap:wrap}
  .cov-btn{padding:3px 8px;border-radius:4px;border:1px solid transparent;font-size:11px;font-weight:500;cursor:pointer;transition:all .12s}
  .cov-btn[data-cov="Full (7yr+)"]{background:var(--tag-full-bg);color:var(--tag-full-tx)}
  .cov-btn[data-cov="Good (4.5-7yr)"]{background:var(--tag-good-bg);color:var(--tag-good-tx)}
  .cov-btn[data-cov="Moderate (2-4.5yr)"]{background:var(--tag-mod-bg);color:var(--tag-mod-tx)}
  .cov-btn[data-cov="Short (1-2yr)"]{background:var(--tag-sht-bg);color:var(--tag-sht-tx)}
  .cov-btn[data-cov="Thin (<1yr)"]{background:var(--tag-thin-bg);color:var(--tag-thin-tx)}
  .cov-btn.active{border-color:currentColor;font-weight:700}
  .cov-btn.inactive{opacity:.4}
  .table-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}
  thead th{position:sticky;top:0;background:var(--surface2);padding:7px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted);border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
  thead th:hover{color:var(--text)}
  thead th.sorted{color:var(--accent)}
  thead th .si{display:inline-block;margin-left:3px;opacity:.5;font-style:normal}
  thead th.sorted .si{opacity:1}
  tbody tr{border-bottom:1px solid var(--border);transition:background .08s}
  tbody tr:hover{background:var(--accent-light)}
  td{padding:6px 12px;vertical-align:middle}
  .crop-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;flex-shrink:0}
  .crop-dot.tomato{background:var(--tomato)}.crop-dot.onion{background:var(--onion)}.crop-dot.potato{background:var(--potato)}
  .mname{font-weight:500}.sname{color:var(--text-muted);font-size:11px}
  .ct{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap}
  .ct.full{background:var(--tag-full-bg);color:var(--tag-full-tx)}
  .ct.good{background:var(--tag-good-bg);color:var(--tag-good-tx)}
  .ct.moderate{background:var(--tag-mod-bg);color:var(--tag-mod-tx)}
  .ct.short{background:var(--tag-sht-bg);color:var(--tag-sht-tx)}
  .ct.thin{background:var(--tag-thin-bg);color:var(--tag-thin-tx)}
  .wb-wrap{display:flex;align-items:center;gap:7px}
  .wb{height:5px;border-radius:3px;min-width:2px;flex-shrink:0}
  .wb.full{background:var(--tag-full-tx)}.wb.good{background:var(--tag-good-tx)}
  .wb.moderate{background:var(--tag-mod-tx)}.wb.short{background:var(--tag-sht-tx)}.wb.thin{background:var(--tag-thin-tx)}
  .dc{color:var(--text-muted);font-size:11px;white-space:nowrap}
  .footer-bar{display:flex;align-items:center;justify-content:space-between;padding:9px 20px;background:var(--surface);border-top:1px solid var(--border);position:sticky;bottom:0;gap:10px;flex-wrap:wrap}
  .rc{font-size:11px;color:var(--text-muted)}
  .pgn{display:flex;gap:3px;align-items:center}
  .pb{padding:3px 8px;border:1px solid var(--border);border-radius:4px;background:var(--surface);color:var(--text);font-size:11px;cursor:pointer;transition:all .1s}
  .pb:hover{background:var(--surface2)}
  .pb.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .pb:disabled{opacity:.35;cursor:not-allowed}
  .pi{font-size:11px;color:var(--text-muted)}
  .empty{text-align:center;padding:40px 20px;color:var(--text-muted)}
  .container{padding:0 0 80px}
</style>
"""

TEMPLATE_BODY = r"""
<div class="header">
  <div class="header-top">
    <span class="header-title">TOP Market Coverage</span>
    <span class="header-subtitle">Tomato &middot; Onion &middot; Potato &nbsp;|&nbsp; 2017&ndash;2024</span>
  </div>
  <div class="summary-row">
    <button class="crop-chip active" data-crop="all">All <span class="count" id="cnt-all"></span></button>
    <button class="crop-chip" data-crop="tomato">Tomato <span class="count" id="cnt-tomato"></span></button>
    <button class="crop-chip" data-crop="onion">Onion <span class="count" id="cnt-onion"></span></button>
    <button class="crop-chip" data-crop="potato">Potato <span class="count" id="cnt-potato"></span></button>
  </div>
  <div class="filter-row">
    <div class="search-wrap">
      <svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="8" cy="8" r="6"/><line x1="13" y1="13" x2="18" y2="18"/>
      </svg>
      <input id="searchInput" type="text" placeholder="Search market or state&hellip;" autocomplete="off">
    </div>
    <select id="stateFilter"><option value="">All States</option></select>
    <div class="cov-filters" id="covFilters">
      <button class="cov-btn" data-cov="Full (7yr+)">Full 7yr+</button>
      <button class="cov-btn" data-cov="Good (4.5-7yr)">Good 4.5&ndash;7yr</button>
      <button class="cov-btn" data-cov="Moderate (2-4.5yr)">Moderate 2&ndash;4.5yr</button>
      <button class="cov-btn" data-cov="Short (1-2yr)">Short 1&ndash;2yr</button>
      <button class="cov-btn" data-cov="Thin (&lt;1yr)">Thin &lt;1yr</button>
    </div>
  </div>
</div>
<div class="container">
  <div class="table-wrap">
    <table id="mainTable">
      <thead><tr>
        <th data-col="market">Market <i class="si">&updownarrow;</i></th>
        <th data-col="state">State <i class="si">&updownarrow;</i></th>
        <th data-col="crop">Crop <i class="si">&updownarrow;</i></th>
        <th data-col="first_week">First Week <i class="si">&updownarrow;</i></th>
        <th data-col="last_week">Last Week <i class="si">&updownarrow;</i></th>
        <th data-col="n_weeks">Weeks <i class="si">&updownarrow;</i></th>
        <th data-col="coverage_category">Coverage <i class="si">&updownarrow;</i></th>
      </tr></thead>
      <tbody id="tableBody"></tbody>
    </table>
    <div class="empty" id="emptyState" style="display:none"><p>No markets match the current filters.</p></div>
  </div>
</div>
<div class="footer-bar">
  <div class="rc" id="resultCount"></div>
  <div class="pgn" id="pagination"></div>
</div>
"""

TEMPLATE_SCRIPT = r"""
const MAX_WEEKS=419,PAGE_SIZE=60;
let filteredData=[],currentCrop='all',currentState='',activeCovs=new Set(),searchTerm='',sortCol='n_weeks',sortDir=-1,currentPage=1;
const covClass={'Full (7yr+)':'full','Good (4.5-7yr)':'good','Moderate (2-4.5yr)':'moderate','Short (1-2yr)':'short','Thin (<1yr)':'thin'};
function updateCropCounts(){
  const c={all:DATA.length,tomato:0,onion:0,potato:0};
  DATA.forEach(r=>c[r.crop]++);
  ['all','tomato','onion','potato'].forEach(k=>document.getElementById('cnt-'+k).textContent=c[k].toLocaleString());
}
function populateStateFilter(){
  const sel=document.getElementById('stateFilter');
  const states=[...new Set((currentCrop==='all'?DATA:DATA.filter(r=>r.crop===currentCrop)).map(r=>r.state))].sort();
  sel.innerHTML='<option value="">All States</option>';
  states.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;if(s===currentState)o.selected=true;sel.appendChild(o);});
}
function applyFilters(){
  let d=DATA;
  if(currentCrop!=='all')d=d.filter(r=>r.crop===currentCrop);
  if(currentState)d=d.filter(r=>r.state===currentState);
  if(activeCovs.size>0)d=d.filter(r=>activeCovs.has(r.coverage_category));
  if(searchTerm){const q=searchTerm.toLowerCase();d=d.filter(r=>r.market.toLowerCase().includes(q)||r.state.toLowerCase().includes(q));}
  d=[...d].sort((a,b)=>{let av=a[sortCol],bv=b[sortCol];if(typeof av==='string'){av=av.toLowerCase();bv=bv.toLowerCase();}return av<bv?-sortDir:av>bv?sortDir:0;});
  filteredData=d;currentPage=1;renderTable();renderPagination();
}
function renderTable(){
  const tbody=document.getElementById('tableBody'),empty=document.getElementById('emptyState');
  const start=(currentPage-1)*PAGE_SIZE,slice=filteredData.slice(start,start+PAGE_SIZE);
  if(filteredData.length===0){tbody.innerHTML='';empty.style.display='';}
  else{
    empty.style.display='none';
    tbody.innerHTML=slice.map(r=>{
      const cls=covClass[r.coverage_category]||'thin';
      const bw=Math.round((r.n_weeks/MAX_WEEKS)*72);
      return `<tr>
        <td><span class="mname">${r.market}</span></td>
        <td><span class="sname">${r.state}</span></td>
        <td><span class="crop-dot ${r.crop}"></span>${r.crop.charAt(0).toUpperCase()+r.crop.slice(1)}</td>
        <td class="dc">${r.first_week}</td>
        <td class="dc">${r.last_week}</td>
        <td><div class="wb-wrap"><div class="wb ${cls}" style="width:${bw}px"></div><span>${r.n_weeks}</span></div></td>
        <td><span class="ct ${cls}">${r.coverage_category}</span></td>
      </tr>`;
    }).join('');
  }
  document.getElementById('resultCount').textContent=filteredData.length.toLocaleString()+' of '+DATA.length.toLocaleString()+' markets';
}
function renderPagination(){
  const total=Math.ceil(filteredData.length/PAGE_SIZE),pg=document.getElementById('pagination');
  if(total<=1){pg.innerHTML='';return;}
  let html=`<button class="pb" onclick="goPage(${currentPage-1})" ${currentPage===1?'disabled':''}>&#8249;</button>`;
  const pages=new Set([1]);
  for(let i=Math.max(2,currentPage-2);i<=Math.min(total-1,currentPage+2);i++)pages.add(i);
  if(total>1)pages.add(total);
  const uniq=[...pages].sort((a,b)=>a-b);
  let prev=0;
  uniq.forEach(p=>{
    if(p-prev>1)html+=`<span class="pi">&hellip;</span>`;
    html+=`<button class="pb ${p===currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;
    prev=p;
  });
  html+=`<button class="pb" onclick="goPage(${currentPage+1})" ${currentPage===total?'disabled':''}>&#8250;</button>`;
  html+=`<span class="pi">&nbsp;${currentPage}/${total}</span>`;
  pg.innerHTML=html;
}
function goPage(p){const total=Math.ceil(filteredData.length/PAGE_SIZE);if(p<1||p>total)return;currentPage=p;renderTable();renderPagination();window.scrollTo({top:0,behavior:'smooth'});}
document.querySelectorAll('.crop-chip').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.crop-chip').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');currentCrop=btn.dataset.crop;currentState='';
  populateStateFilter();applyFilters();
}));
document.getElementById('stateFilter').addEventListener('change',e=>{currentState=e.target.value;applyFilters();});
let st;document.getElementById('searchInput').addEventListener('input',e=>{clearTimeout(st);st=setTimeout(()=>{searchTerm=e.target.value.trim();applyFilters();},180);});
document.querySelectorAll('.cov-btn').forEach(btn=>btn.addEventListener('click',()=>{
  const cov=btn.dataset.cov;
  if(activeCovs.has(cov)){activeCovs.delete(cov);btn.classList.remove('active');}
  else{activeCovs.add(cov);btn.classList.add('active');}
  document.querySelectorAll('.cov-btn').forEach(b=>b.classList.toggle('inactive',activeCovs.size>0&&!activeCovs.has(b.dataset.cov)));
  applyFilters();
}));
document.querySelectorAll('thead th[data-col]').forEach(th=>th.addEventListener('click',()=>{
  const col=th.dataset.col;
  if(sortCol===col)sortDir*=-1;else{sortCol=col;sortDir=1;}
  document.querySelectorAll('thead th').forEach(t=>{t.classList.remove('sorted');t.querySelector('.si').innerHTML='&updownarrow;';});
  th.classList.add('sorted');th.querySelector('.si').textContent=sortDir===1?'↑':'↓';
  applyFilters();
}));
updateCropCounts();populateStateFilter();applyFilters();
"""

html = (
    TEMPLATE_TOP
    + TEMPLATE_BODY
    + '<script>\n'
    + 'const DATA = '
    + raw
    + ';\n'
    + TEMPLATE_SCRIPT
    + '\n</script>'
)

out = 'data/market_coverage_browser.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

print('Written: ' + out)
print('Size: ' + str(round(os.path.getsize(out)/1024, 1)) + ' KB')
