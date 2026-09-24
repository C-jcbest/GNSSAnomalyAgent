"use strict";
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={hampel:'Hampel',cusum:'CUSUM',reference_trend:'稳健基准趋势',iforest:'Isolation Forest',visual:'视觉模型',daily_union:'同期视觉∪Hampel',fixed:'固定数值＋视觉',split_fusion:'独立分工组合',split_fusion_evidence:'证据约束分工组合',rule:'规则式选择',generic:'通用 Agent',lma:'LMA',lma_no_vision:'LMA 去视觉',lma_no_review:'LMA 去复核'};
const kinds={spike:'尖峰',step:'台阶',drift:'缓慢增长',acceleration:'加速增长',variance:'波动增大',normal:'未注入'};
const states={native:'原生缺测',complete:'完整',random10:'随机缺测 10%',random25:'随机缺测 25%',block25:'连续缺测 25%'};
const semantics={quality:'质量形态（注入）',deformation:'位移形态（注入）',undetermined:'成因未定',normal:'未注入'};
const splits={development:'开发集',validation:'验证集',test:'测试集'};
const statusNames={ok:'完成',error:'技术失败',insufficient:'数据不足'};
const numericalMethods=new Set(['hampel','cusum','iforest','reference_trend']);
const methodScope=methods=>[methods.some(method=>numericalMethods.has(method))?'数值':null,methods.includes('visual')?'视觉':null,methods.some(method=>!numericalMethods.has(method)&&method!=='visual')?'组合/策略':null].filter(Boolean).join(' / ');
const chartNames={overview:'总体检测表现',heatmap:'类型 / 状态能力热图',missing:'缺测条件对比',efficiency:'效果与耗时',ablation:'LMA 消融比较'};
const metricNames={f1:'F1',precision:'Precision',recall:'Recall',completion_rate:'有效完成率',first_attempt_completion_rate:'首尝试完成率',numerical_coverage:'数值覆盖率',normal_false_alarm_rate_completed:'未注入窗报告率（非现场误报）'};
let runs=[],selected=null,detail=null,runVersion=0,caseVersion=0,casePage=0,currentCase=null,currentCaseKey=null,labelGallerySupported=false;
let labelVersion=0,labelLoaded=false,labelCases=[],labelData=new Map(),labelRange='focus';
const num=(v,d=3)=>v===null||v===undefined?'—':Number(v).toFixed(d);
const pct=v=>v===null||v===undefined?'—':(100*v).toFixed(1)+'%';
const time=v=>v?new Date(v).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}):'未记录';
const url=(path,params)=>path+'?'+new URLSearchParams(Object.entries(params).filter(([,v])=>v!==null&&v!==undefined));
function error(e){$('alert').hidden=false;$('alert').textContent=e.message||String(e);}
async function api(path){const r=await fetch(path,{cache:'no-store'});const data=await r.json();if(!r.ok)throw new Error(data.error||'读取失败');return data;}
const options=(map,blank='')=>(blank?`<option value="">${esc(blank)}</option>`:'')+Object.entries(map).map(([k,v])=>`<option value="${esc(k)}">${esc(v)}</option>`).join('');
const stat=(label,value,unit)=>`<div class="stat"><small>${esc(label)}</small><strong>${esc(value)}<span>${esc(unit)}</span></strong></div>`;
function exportsHTML(params){return ['png','svg','pdf'].map(f=>`<a href="${url('/export',{...params,format:f})}">${f.toUpperCase()} ↓</a>`).join('');}
async function refresh(){
  $('alert').hidden=true;$('refresh').disabled=true;
  try{const data=await api('/api/runs');runs=data.runs;labelGallerySupported=Array.isArray(data.capabilities)&&data.capabilities.includes('injection_label_gallery_v1');
    if(runs.length&&!runs.some(r=>Number.isInteger(r.sampling_hours)))throw new Error('服务进程仍运行旧版后端：请重启 gnss-exp serve 后刷新，旧接口缺少采样间隔。');
    const daily=runs.filter(r=>r.sampling_hours===24);
    $('archive-stats').innerHTML=stat('日序列运行',daily.length,'份档案')+stat('日运行已落盘',daily.reduce((n,r)=>n+r.present,0),'条方法结果')+stat('日实验方法',new Set(daily.flatMap(r=>r.methods)).size,'个配置')+stat('记录齐全',daily.filter(r=>r.complete).length,'份 · 非效果评级');
    renderHistory();if(data.errors.length)error(new Error(data.errors.map(e=>e.name+'：'+e.message).join('；')));
    const visible=runs.filter(r=>$('include-hourly').checked||r.sampling_hours===24);
    if(visible.length){const chosen=visible.find(r=>r.key===selected)||visible.find(r=>!r.limited)||visible[0];await openRun(chosen.key);}else{$('run-detail').innerHTML='<div class="empty">尚无每日 15:00 的运行记录。可勾选左侧开关查看旧小时档案。</div>';}
  }catch(e){error(e);}finally{$('refresh').disabled=false;}
}
function renderHistory(){
  const query=$('search').value.toLowerCase(),split=$('split').value;
  const filtered=runs.filter(r=>($('include-hourly').checked||r.sampling_hours===24)&&(!split||r.split===split)&&r.name.toLowerCase().includes(query));
  $('run-count').textContent=filtered.length;
  $('run-list').innerHTML=filtered.map(r=>`<button class="run-item ${r.key===selected?'selected':''}" data-run="${r.key}" aria-pressed="${r.key===selected}"><span class="run-name">${esc(r.name)}</span><span class="run-meta"><span>${esc(time(r.created_at))}</span><span>${esc(splits[r.split]||r.split)}</span></span><span class="run-tags"><span class="badge">${r.sampling_hours===24?'每日15点':'小时归档'}</span><span class="badge ${r.kind==='synthetic_demo_only'?'demo':''}">${r.kind==='synthetic_demo_only'?'工程示例':r.kind==='platform_pilot_provisional'?'开发预实验':r.kind==='daily_injection_unverified_v1'?'日注入 · 背景未认证':'受控注入'}</span><span class="badge ${r.complete&&!r.failures?'success':''}">${r.limited?'限量 · ':''}${r.present}/${r.expected}</span></span></button>`).join('')||'<div class="empty">没有匹配的记录</div>';
  document.querySelectorAll('[data-run]').forEach(b=>b.onclick=()=>openRun(b.dataset.run));
}
async function openRun(key){
  selected=key;renderHistory();const version=++runVersion;++caseVersion;++labelVersion;currentCase=null;currentCaseKey=null;labelLoaded=false;labelData=new Map();labelRange='focus';
  $('run-detail').innerHTML='<div class="empty">正在整理运行记录…</div>';
  try{const d=await api(url('/api/run',{run:key}));if(version!==runVersion)return;detail=d;renderDetail();}catch(e){if(version===runVersion){$('run-detail').innerHTML='<div class="empty">读取失败，请刷新或选择其他记录。</div>';error(e);}}
}
function renderDetail(){
  const d=detail,m=d.meta,s=d.summary;
  const all=s.summary.filter(r=>r.dimension==='all');
  const charts=all.some(r=>r.method==='lma')?chartNames:Object.fromEntries(Object.entries(chartNames).filter(([key])=>key!=='ablation'));
  const hasLabelledRun=m.dataset_kind==='daily_injection_unverified_v1'&&m.sampling_hours===24;
  const hasLabels=hasLabelledRun&&labelGallerySupported;
  $('run-detail').innerHTML=`<div class="detail-title"><div><p class="eyebrow">RUN / ${esc(m.run_id.slice(0,12))}</p><h2>${esc(d.name)}</h2><p class="detail-sub">${esc(splits[m.split])} · ${all.length} 个方法配置 · ${s.records_present} / ${s.records_expected} 条记录<br>数据版本 <code>${esc(m.dataset_sha256.slice(0,16))}</code></p></div><div class="exports"><a href="${url('/metrics.csv',{run:selected})}">指标 CSV ↓</a><a class="primary" href="${url('/paper.zip',{run:selected})}">论文图包 ↓</a></div></div>
  <div class="run-context" aria-label="本次实验内容"><div><small>输入与范围</small><strong>${m.sampling_hours===24?'每日 15:00':'历史小时输入'}</strong><span>${esc(splits[m.split]||m.split)} · ${new Set(d.cases.map(r=>r.case_id)).size} 个案例</span></div><div><small>方法与重复</small><strong>${all.length} 个配置</strong><span>${esc(methodScope(all.map(r=>r.method)))} · 模型配置重复 ${esc(m.repeats)} 次</span></div><div><small>记录状态</small><strong>${s.records_present} / ${s.records_expected}</strong><span>${s.all_records_present?'记录齐全':'记录未齐'} · 指标 ${esc(s.metric_version)}</span></div></div>
  ${m.sampling_hours===24?'':'<div class="notice">此为历史小时运行，仅供追溯；不要与每日15点结果混合排序。</div>'}
  ${m.dataset_kind==='daily_injection_unverified_v1'?'<div class="notice">本轮仅有受控注入标签：F1为注入掩码一致度，原生变化未标注；未注入窗的报告不等于现场误报。组合行的请求和耗时只计额外执行增量，端到端成本须计入数值与视觉两路。</div>':''}
  ${hasLabelledRun&&!labelGallerySupported?'<div class="notice">当前服务后端尚不支持注入标注图谱。请重启服务后刷新；指标与案例仍可查看。</div>':''}
  ${d.notices.length?`<div class="notice">${d.notices.map(esc).join('<br>')}</div>`:''}
  <div class="tabs" role="tablist" aria-label="运行详情"><button role="tab" aria-selected="true" class="active" data-tab="metrics">指标与图表</button>${hasLabels?'<button role="tab" aria-selected="false" data-tab="labels">注入标注</button>':''}<button role="tab" aria-selected="false" data-tab="cases">案例证据</button><button role="tab" aria-selected="false" data-tab="config">运行配置</button></div>
  <div id="metrics-tab" class="tab-panel"><div class="panel-heading"><h3>方法能力概览</h3><span class="hint">仅比较同一运行、同一数据与指标版本</span></div><div class="filters"><label>图表<select id="chart-type">${options(charts)}</select></label><label>指标<select id="chart-metric">${options(metricNames)}</select></label><label>图中分组<select id="chart-dimension">${options({kind:'异常类型',state:'缺测状态',semantic:'注入形态分组'})}</select></label></div><div class="methods" id="methods">${all.map(r=>`<label><input type="checkbox" value="${esc(r.method)}" checked>${esc(names[r.method]||r.method)}</label>`).join('')}</div><div class="panel-heading"><p class="hint" id="chart-status" aria-live="polite">正在生成图表…</p><div class="exports" id="figure-exports"></div></div><div class="chart-wrap"><img id="main-chart" class="chart" alt="方法性能比较图"></div><p class="hint chart-note">图中误差条为模型重复标准差，不是置信区间；数值方法通常只运行一次。N/A 不是零分，缺测位置不参与定位评分。</p><details class="metric-help"><summary>指标速读 · F1、未注入报警与完成率分别说明什么？</summary><div class="metric-grid"><p><strong>Precision / Recall / F1</strong>逐个已观测日期×分量匹配注入掩码；F1 为精确率和召回率的调和平均。无可观测阳性时 F1 为 N/A。</p><p><strong>未注入窗报警率</strong>完成的未注入窗口里至少报出一个位置的比例；背景未认证正常，不等于现场误报率。</p><p><strong>最终 / 首尝试完成率</strong>前者包括允许的重试；后者只看模型首次 HTTP 响应是否合法。失败、数据不足需同时阅读。</p><p><strong>数值覆盖率 / 成本</strong>覆盖率只代表工具实际计算的位置；固定组合仅指其数值部分。费用缺单价时未知，不是零。</p></div></details><h3>总体指标明细</h3><div id="metrics-table" class="table-scroll"></div><p class="hint">数值方法 1 次，模型方法按配置重复；均值 ± SD 仅描述同输入的重复波动。完成率、报告负担和用量需与 F1 同时阅读；历史缺少的诊断显示—。</p><div class="panel-heading breakdown-heading"><h3>分组指标</h3><label>按 <select id="breakdown-dimension">${options({kind:'异常类型',state:'缺测状态',semantic:'注入形态分组'})}</select> 查看</label></div><div id="breakdown-table" class="table-scroll"></div><p class="hint">分组行是同一运行中的描述，不是独立站点样本；未注入组无阳性时 F1 为 N/A，请查看报警率。</p></div>
  ${hasLabels?'<div id="labels-tab" class="tab-panel" hidden><div class="panel-heading"><h3>受控注入标注图谱</h3><span class="hint">同一来源窗 · 六种情景</span></div><p class="notice">色带与加粗曲线仅表示注入目标，不是原始观测的现场异常真值；灰带是原生缺测。“未注入”不等于现场正常。此视图只读取冻结标签，不参与检测。</p><div class="label-controls"><label>来源窗<select id="label-source"></select></label><label>缺测状态<select id="label-state"></select></label><div class="label-segments" role="group" aria-label="标注图范围"><button type="button" data-label-range="focus" class="active" aria-pressed="true">聚焦标注</button><button type="button" data-label-range="all" aria-pressed="false">180天全窗</button></div></div><div id="label-jump" class="label-jump"></div><div id="label-gallery" aria-live="polite"></div></div>':''}
  <div id="cases-tab" class="tab-panel" hidden><h3>对照曲线与检测证据</h3><p class="hint">绿色底带是真值，橙色顶带是预测；灰色区域为缺测。真值仅用于本页评测展示。</p><div class="filters"><label>方法<select id="case-method">${options(Object.fromEntries(all.map(r=>[r.method,names[r.method]])),'全部')}</select></label><label>类型<select id="case-kind">${options(kinds,'全部')}</select></label><label>缺测<select id="case-state">${options(states,'全部')}</select></label><label>状态<select id="case-status">${options(statusNames,'全部')}</select></label><label>案例<input id="case-search" type="search" placeholder="输入案例 ID"></label></div><div class="filters"><label>记录<select id="case-record" aria-label="选择案例记录"></select></label><div class="paging"><button id="case-prev" class="quiet">上一页</button><span id="case-count" class="hint"></span><button id="case-next" class="quiet">下一页</button></div></div><div id="case-body"></div></div>
  <div id="config-tab" class="tab-panel" hidden><h3>可追溯的运行配置</h3><p class="hint">不同数据版本、划分或案例范围的结果不可直接排序。本页展示运行时保存的配置。</p><pre class="run-config" id="run-config"></pre></div>`;
  $('run-config').textContent=JSON.stringify(m,null,2);
  $('metrics-table').innerHTML='<table><thead><tr><th>方法</th><th>Precision</th><th>Recall</th><th>F1 ± SD</th><th>最终完成率</th><th>首尝试完成率</th><th>数值覆盖率</th><th>未注入窗报警率</th><th>秒 / 例</th><th>模型请求 / 例</th><th>重复</th></tr></thead><tbody>'+all.map(r=>`<tr><td>${esc(names[r.method]||r.method)}</td><td>${num(r.precision_mean)}</td><td>${num(r.recall_mean)}</td><td class="score">${r.f1_mean===null?'—':num(r.f1_mean)+' ± '+num(r.f1_std)}</td><td>${pct(r.completion_rate_mean)}</td><td>${pct(r.first_attempt_completion_rate_mean)}</td><td>${pct(r.numerical_coverage_mean)}</td><td>${pct(r.normal_false_alarm_rate_completed_mean)}</td><td>${num(r.mean_seconds_mean,2)}</td><td>${num(r.mean_model_requests_mean,1)}</td><td>${r.repeats}</td></tr>`).join('')+'</tbody></table>';
  $('breakdown-dimension').onchange=renderBreakdown;renderBreakdown();
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-tab]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-selected',x===b);});['metrics','labels','cases','config'].forEach(t=>{if($(t+'-tab'))$(t+'-tab').hidden=t!==b.dataset.tab;});if(b.dataset.tab==='labels'&&!labelLoaded){labelLoaded=true;renderLabelGallery();}if(b.dataset.tab==='cases'&&!currentCaseKey)filterCases();});
  ['chart-type','chart-metric','chart-dimension'].forEach(id=>$(id).onchange=updateChart);document.querySelectorAll('#methods input').forEach(i=>i.onchange=updateChart);
  ['case-method','case-kind','case-state','case-status','case-search'].forEach(id=>$(id).oninput=()=>{casePage=0;filterCases();});
  $('case-record').onchange=()=>openCase($('case-record').value);$('case-prev').onclick=()=>{casePage--;filterCases();};$('case-next').onclick=()=>{casePage++;filterCases();};casePage=0;updateChart();
  if(hasLabels)initLabelGallery();
}
function initLabelGallery(){
  labelCases=[...new Map(detail.cases.map(row=>[row.case_id,row])).values()];
  const sources=[...new Set(labelCases.map(labelSource))];
  const availableStates=[...new Set(labelCases.map(row=>row.state))];
  $('label-source').innerHTML=sources.map((source,index)=>`<option value="${esc(source)}">来源窗 ${index+1} · ${esc(source.slice(0,8))}</option>`).join('');
  $('label-state').innerHTML=Object.entries(states).filter(([state])=>availableStates.includes(state)).map(([state,name])=>`<option value="${esc(state)}">${esc(name)}</option>`).join('');
  if(availableStates.includes('native'))$('label-state').value='native';
  $('label-source').onchange=renderLabelGallery;$('label-state').onchange=renderLabelGallery;
  document.querySelectorAll('[data-label-range]').forEach(button=>button.onclick=()=>{labelRange=button.dataset.labelRange;document.querySelectorAll('[data-label-range]').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',item===button);});renderLabelGallery();});
}
function labelBounds(events,length){
  if(labelRange==='all'||!events.length)return [0,length-1];
  const start=Math.min(...events.map(event=>event.start)),end=Math.max(...events.map(event=>event.end));
  return [Math.max(0,start-14),Math.min(length-1,end+14)];
}
const labelSource=row=>row.source_group_id??row.group??row.case_id;
function renderLabelGallery(){
  const version=++labelVersion,run=selected,source=$('label-source').value,state=$('label-state').value;
  const byKind=new Map(labelCases.filter(row=>labelSource(row)===source&&row.state===state).map(row=>[row.kind,row]));
  $('label-jump').innerHTML=Object.entries(kinds).map(([kind,name])=>`<button type="button" data-jump="${kind}" ${byKind.has(kind)?'':'disabled'}>${esc(name)}</button>`).join('');
  $('label-gallery').innerHTML=Object.entries(kinds).map(([kind,name])=>`<section class="label-example" id="label-kind-${kind}"><div class="label-heading"><div><span class="label-swatch kind-${kind}"></span><h4>${esc(name)}</h4><span class="label-channel" id="label-channel-${kind}">${byKind.has(kind)?'读取标签…':'本运行无此情景'}</span></div><span class="label-index" id="label-index-${kind}"></span></div><p class="label-period" id="label-period-${kind}"></p>${byKind.has(kind)?`<img class="label-chart" id="label-chart-${kind}" alt="${esc(name)}的三轴观测与注入标注" loading="lazy"><div class="exports" id="label-export-${kind}"></div>`:''}</section>`).join('');
  document.querySelectorAll('[data-jump]').forEach(button=>button.onclick=()=>$('label-kind-'+button.dataset.jump).scrollIntoView({behavior:'smooth',block:'start'}));
  for(const [kind,row] of byKind){
    const load=labelData.has(row.case_id)?Promise.resolve(labelData.get(row.case_id)):api(url('/api/label',{run,result:row.result_key}));
    load.then(data=>{
      if(version!==labelVersion||run!==selected)return;
      labelData.set(row.case_id,data);
      const events=data.truth.events,length=data.window.values.length;
      const date=index=>new Date(data.window.timestamps[index]).toLocaleDateString('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'});
      const cells=new Set();
      events.forEach(event=>event.channels.forEach(channel=>{const axis=['N','E','U'].indexOf(channel);for(let index=event.start;index<=event.end;index++)if(data.window.values[index][axis]!==null)cells.add(index+'-'+channel);}));
      $('label-channel-'+kind).textContent=events.length?`${[...new Set(events.flatMap(event=>event.channels))].join(' / ')} 分量 · ${cells.size} 个已观测标注单元`:'无新增注入标签';
      $('label-index-'+kind).textContent=events.length?`${Math.min(...events.map(event=>event.start))}–${Math.max(...events.map(event=>event.end))} · 索引从0起`:'0 个注入目标';
      $('label-period-'+kind).textContent=events.length?events.map(event=>`${event.channels.join('/')}：${date(event.start)} 至 ${date(event.end)}（索引 ${event.start}–${event.end}）`).join('；'):'当前案例没有额外注入，原生变化仍未获得逐日真值。';
      const [start,end]=labelBounds(events,length),params={run,chart:'labels',result:row.result_key,start,end};
      const image=$('label-chart-'+kind);image.src=url('/chart',params);image.onerror=()=>{if(version===labelVersion)$('label-period-'+kind).textContent='图表加载失败，请刷新重试。';};
      $('label-export-'+kind).innerHTML=['png','svg'].map(format=>`<a href="${url('/export',{...params,format})}">${format.toUpperCase()} ↓</a>`).join('');
    }).catch(e=>{if(version===labelVersion)$('label-period-'+kind).textContent='标签读取失败：'+(e.message||String(e));});
  }
}
function renderBreakdown(){
  const dimension=$('breakdown-dimension').value;
  const labels=dimension==='kind'?kinds:dimension==='state'?states:semantics;
  const rows=detail.summary.summary.filter(r=>r.dimension===dimension);
  $('breakdown-table').innerHTML=rows.length?'<table><thead><tr><th>方法</th><th>分组</th><th>每次案例</th><th>Precision</th><th>Recall</th><th>F1 ± SD</th><th>完成率</th><th>未注入窗报警率</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${esc(names[r.method]||r.method)}</td><td>${esc(labels[r.value]||r.value)}</td><td>${r.cases_per_repeat}</td><td>${num(r.precision_mean)}</td><td>${num(r.recall_mean)}</td><td class="score">${r.f1_mean===null?'—':num(r.f1_mean)+' ± '+num(r.f1_std)}</td><td>${pct(r.completion_rate_mean)}</td><td>${pct(r.normal_false_alarm_rate_completed_mean)}</td></tr>`).join('')+'</tbody></table>':'<div class="empty">此运行没有该分组的结果</div>';
}
function updateChart(){
  const methods=[...document.querySelectorAll('#methods input:checked')].map(x=>x.value);
  const chart=$('chart-type').value;
  $('chart-metric').disabled=['overview','efficiency'].includes(chart);$('chart-dimension').disabled=chart!=='heatmap';
  if(!methods.length){$('main-chart').hidden=true;$('figure-exports').innerHTML='';$('chart-status').textContent='请至少选择一种方法';return;}
  const params={run:selected,chart,metric:$('chart-metric').value,dimension:$('chart-dimension').value,methods:methods.join(',')};
  const image=$('main-chart');image.hidden=false;$('chart-status').textContent='正在生成图表…';image.onload=()=>{$('chart-status').textContent='可导出 300 dpi PNG 或矢量 SVG / PDF';};image.onerror=()=>{$('chart-status').textContent='图表生成失败，请刷新重试';};image.src=url('/chart',params);image.alt=chartNames[chart];$('figure-exports').innerHTML=exportsHTML(params);
}
function filterCases(){
  const criteria={method:$('case-method').value,kind:$('case-kind').value,state:$('case-state').value,status:$('case-status').value};
  const rows=detail.cases.filter(r=>Object.entries(criteria).every(([k,v])=>!v||r[k]===v)&&r.case_id.includes($('case-search').value.trim()));
  const start=casePage*100,page=rows.slice(start,start+100);
  $('case-count').textContent=rows.length?`${start+1}–${start+page.length} / ${rows.length}`:'0 条';$('case-prev').disabled=casePage===0;$('case-next').disabled=start+100>=rows.length;
  $('case-record').innerHTML=page.map(r=>`<option value="${r.result_key}">${esc(r.case_id.slice(0,12))} · ${esc(names[r.method])} · ${esc(kinds[r.kind])} / ${esc(states[r.state])} · R${r.repeat+1} · ${esc(statusNames[r.status])}</option>`).join('');
  if(page.length)openCase(page[0].result_key);else{++caseVersion;currentCaseKey=null;$('case-body').innerHTML='<div class="empty">没有符合筛选条件的案例</div>';}
}
async function openCase(key){
  const version=++caseVersion,run=selected;currentCaseKey=key;$('case-body').innerHTML='<div class="empty">正在校验数据与标签并还原曲线…</div>';
  try{const c=await api(url('/api/case',{run,result:key}));if(version!==caseVersion||run!==selected)return;currentCase=c;
    const n=c.window.values.length,mt=c.metrics;
    $('case-body').innerHTML=`<div class="case-metrics"><span>TP<b>${mt.tp}</b></span><span>FP<b>${mt.fp}</b></span><span>FN<b>${mt.fn}</b></span><span>隐藏事件<b>${mt.hidden_events}</b></span><span>状态 <b>${esc(statusNames[c.prediction.status])}</b></span></div><div class="zoom"><label>起点 <input id="zoom-start" type="number" min="0" max="${n-2}" value="0"></label><label>终点 <input id="zoom-end" type="number" min="1" max="${n-1}" value="${n-1}"></label><button class="quiet" id="zoom-apply">查看区间</button><button class="quiet" id="zoom-reset">完整窗口</button><span id="zoom-error" class="hint error"></span></div><div class="exports" id="case-export"></div><img id="case-chart" class="chart" alt="三轴位移与真值预测叠图"><details><summary>查看预测依据、工具轨迹与调用用量</summary><pre id="case-trace" class="trace"></pre></details>`;
    $('case-trace').textContent=JSON.stringify({prediction:c.prediction,metrics:c.metrics,diagnostics:c.diagnostics,tools:c.tools,model_calls:c.model_calls},null,2);
    $('zoom-apply').onclick=updateCaseChart;$('zoom-reset').onclick=()=>{$('zoom-start').value=0;$('zoom-end').value=n-1;updateCaseChart();};updateCaseChart();
  }catch(e){if(version===caseVersion){$('case-body').innerHTML=`<div class="empty error">${esc(e.message)}</div>`;}}
}
function updateCaseChart(){
  const start=Number($('zoom-start').value),end=Number($('zoom-end').value),n=currentCase.window.values.length;
  if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end>=n||start>=end){$('zoom-error').textContent='请选择有效的起止索引，起点须小于终点。';return;}
  $('zoom-error').textContent='';const params={run:selected,chart:'case',result:currentCaseKey,start,end};$('case-chart').src=url('/chart',params);$('case-export').innerHTML=exportsHTML(params);
}
async function showData(){
  try{const data=await api(url('/api/data',{view:$('data-version').value})),items=data.snapshots,inventory=data.inventory,collection=data.collection;
    const prep=data.preparation,daily=prep?.sampling_hours===24&&$('data-version').value!=='raw';
    const size=r=>r.audit.expected_days??r.audit.expected_hours;
    const coverageNote=r=>daily&&prep.stations.find(x=>x.station_alias===r.alias)?.status==='low_window_coverage'?'<br><small>长窗完整率不足80%</small>':'';
    items.sort((a,b)=>Number(Boolean(b.reference))-Number(Boolean(a.reference))||size(b)-size(a));
    $('data-stats').innerHTML=stat('本地快照',items.length,'份 · 原值及缺口保留')+stat('当前采样',daily?'每日 15:00':'小时观测','北京时间')+stat('长期窗口',daily?`${prep.window_days} 天`:'120 天','按平台来源分组')+stat('已准备窗口',prep?.long_windows??'—','个 · 未标注候选');
    $('preparation-note').hidden=!prep;
    if(prep){$('preparation-note').textContent=daily?`已保存 ${prep.retained_stations} 站的每日15点全历史，共 ${prep.observed_days} 个有效站点日；每站最近 ${prep.window_days} 天中三轴完整率≥80%的区间形成候选窗口，共 ${prep.long_windows} 个，辅助案例 ${prep.auxiliary_cases} 个。历史不足的站点仅保留全历史，不填充、不缩短主窗口。尚未生成真值；来源划分保持不变。`:`小时历史归档：${prep.retained_stations} 站，长窗 ${prep.long_windows} 个。当前研究采用每日15点版本；此视图用于追溯。`;}
    $('snapshots').innerHTML='<table><thead><tr><th>站点别名</th><th>起点（北京时间）</th><th>终点（不含）</th><th>有效 / 网格点</th><th>区间内缺测</th><th>来源划分</th><th>操作</th></tr></thead><tbody>'+items.map(r=>`<tr><td>${esc(r.reference?.name||r.alias)}${r.reference?' · 真实趋势线索':''}</td><td>${esc(r.start.slice(0,16).replace('T',' '))}</td><td>${esc(r.end.slice(0,16).replace('T',' '))}</td><td>${r.audit.observed_timestamps} / ${size(r)} ${r.sampling_hours===24?'天':'小时'}${r.sampling_hours===24&&size(r)<prep.window_days?'<br><small>不足主窗口长度</small>':''}${coverageNote(r)}</td><td>${pct(r.audit.missing_fraction)}</td><td>${r.reference?'辅助案例':esc(splits[r.split]||'未划分')}</td><td><button class="quiet" data-snapshot="${r.key}" ${size(r)<2||r.audit.observed_timestamps===0?'disabled':''}>查看曲线</button></td></tr>`).join('')+'</tbody></table>';
    document.querySelectorAll('[data-snapshot]').forEach(b=>b.onclick=()=>openSnapshot(items.find(r=>r.key===b.dataset.snapshot)));
  }catch(e){error(e);}
}
function openSnapshot(row){
  const isDaily=row.sampling_hours===24,n=row.audit.expected_days??row.audit.expected_hours,reference=row.reference,step=(row.sampling_hours||1)*3600000;
  $('snapshot-preview').hidden=false;
  $('snapshot-preview').innerHTML=`<h2>${esc(reference?.name||row.alias)} · ${isDaily?'每日15点':'小时归档'}</h2>${reference?`<p class="notice">${esc(reference.source)}：${esc(reference.boundary_status)}。${esc(reference.note)}</p>`:''}<div class="filters"><label>时间范围<select id="snapshot-range">${options({all:'全部可用历史',recent180:'最近 180 天',recent120:'最近 120 天',recent30:'最近 30 天',...(reference?{reference:'SCWM-04 近 4 个月线索区间'}:{})})}</select></label></div><p class="hint">${isDaily?'严格选取北京时间15:00；不取平均、不替换时刻、不补缺口。最近180天用于长期窗口，全历史可保留更早对照。':'原始小时归档，仅用于追溯；当前日序列请切换至收束后的实验数据。'}</p><div id="snapshot-export" class="exports"></div><img id="snapshot-image" class="chart" alt="真实快照三轴位移曲线">`;
  const update=()=>{
    const range=$('snapshot-range').value;let start=0,end=n-1;
    if(range.startsWith('recent'))start=Math.max(0,n-Number(range.slice(6))*24/(row.sampling_hours||1));
    if(range==='reference'){start=Math.max(0,Math.ceil((new Date(reference.start)-new Date(row.start))/step));end=Math.min(n-1,Math.ceil((new Date(reference.end_exclusive)-new Date(row.start))/step)-1);}
    if(start>=end){error(new Error('此区间不足两个采样点，请选择全部历史。'));return;}
    const params={chart:'snapshot',snapshot:row.key,start,end};$('snapshot-image').src=url('/chart',params);$('snapshot-export').innerHTML=exportsHTML(params);
  };
  if(!reference&&n>=180*24/(row.sampling_hours||1))$('snapshot-range').value='recent180';
  $('snapshot-range').onchange=update;update();$('snapshot-preview').scrollIntoView({behavior:'smooth',block:'start'});
}
function markdown(text){
  // Small local document renderer: escape first, no raw HTML, scripts or external links.
  const inline=t=>esc(t).replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  let html='',code=false,table=false,list=false;
  for(const line of text.split('\n')){
    if(line.startsWith('```')){if(list){html+='</ul>';list=false;}html+=code?'</pre>':'<pre>';code=!code;continue;}
    if(code){html+=esc(line)+'\n';continue;}
    if(line.startsWith('|')){if(!table){html+='<div class="table-scroll"><table>';table=true;}if(!/^\|[\s:|\-]+\|$/.test(line))html+='<tr>'+line.split('|').slice(1,-1).map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>';continue;}
    if(table){html+='</table></div>';table=false;}
    if(line.startsWith('- ')){if(!list){html+='<ul>';list=true;}html+='<li>'+inline(line.slice(2))+'</li>';continue;}
    if(list){html+='</ul>';list=false;}
    const heading=line.match(/^(#{1,3}) (.*)/);html+=heading?`<h${Math.min(3,heading[1].length+1)}>${inline(heading[2])}</h${Math.min(3,heading[1].length+1)}>`:line?'<p>'+inline(line)+'</p>':'';
  }
  return html+(table?'</table></div>':'')+(list?'</ul>':'')+(code?'</pre>':'');
}
async function page(name){
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b.dataset.page===name));['history','data','criteria'].forEach(p=>$(p+'-page').hidden=p!==name);
  window.scrollTo(0,0);
  if(name==='data')await showData();if(name==='criteria'){try{$('criteria-content').innerHTML=markdown((await api('/api/criteria')).text);}catch(e){error(e);}}
}
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>page(b.dataset.page));$('refresh').onclick=async()=>{await refresh();if(!$('data-page').hidden)await showData();};$('data-version').onchange=()=>{$('snapshot-preview').hidden=true;showData();};$('search').oninput=renderHistory;$('split').onchange=renderHistory;$('include-hourly').onchange=()=>{renderHistory();if(!$('include-hourly').checked&&selected&&runs.find(r=>r.key===selected)?.sampling_hours!==24){const daily=runs.find(r=>r.sampling_hours===24);if(daily)openRun(daily.key);else{selected=null;detail=null;$('run-detail').innerHTML='<div class="empty">尚无每日 15:00 的运行记录。</div>';}}else if(!selected&&runs.length){const visible=runs.find(r=>$('include-hourly').checked||r.sampling_hours===24);if(visible)openRun(visible.key);}};refresh();
