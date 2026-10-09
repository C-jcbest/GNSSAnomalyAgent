"use strict";

const methods = {
  visual_model: "视觉定位＋模型阶段判别",
  numeric_model: "数值定位＋模型阶段判别",
  visual_model_raw: "消融：阶段仅看原始图",
  visual_model_ungated: "消融：解除阶段活动约束",
  single: "单次视觉：同时定位与判阶段",
  "reference-stage": "参考活动范围＋模型阶段判别",
  "locate-global": "定位对照：全局原图",
  "locate-local": "定位对照：全局＋局部原图",
  "stage-global": "阶段对照：原图＋辅助图",
  "stage-local": "阶段对照：原图＋辅助图＋局部图",
};
const localMethods = ["locate-global", "locate-local", "stage-global", "stage-local"];
const caseNotes = {
  case_0002: "多段活动与边界",
  case_0003: "弱位移漏检 / 平台",
  case_0006: "活动后平台误报",
  case_0007: "阶段混淆 / 越界",
  case_0015: "静稳负对照",
  case_0023: "端点冲突 / 尾部待核",
};
const reviewNotes = {
  case_0002: "原始图有三段较清楚的累积位移。复核时重点看活动起止、末端截断和缺测边界；辅助窗口在边缘不能单独决定活动起点。",
  case_0003: "既有复核发现：视觉定位把两段运动之间的平台合并，且漏掉第 775–855 日的弱位移参考段。该弱段 N 分量首尾 14 日中位数相差约 +4.04 mm，在局部图中较易查看。全局图的较宽纵轴可能压低弱趋势的可见性；这仍是待对照实验检验的因素。",
  case_0006: "既有复核发现：约第 550–700 日的活动后平台被合并进活动范围。判读时先看原始位移是否仍在累积，不能因平台处在较高位移水平就延续活动。参考范围诊断中，第 440–510 日被整体判为 A，而辅助速度先升后降，需重新核对阶段边界。",
  case_0007: "当前视觉定位覆盖参考活动范围，但后半段参考匀速日仍被模型判为加速。已有较大速度不等于仍在加速。另一个数值门控配置在第 516–519 日跨出确认范围，属于独立的契约失败。",
  case_0015: "这是当前评价组唯一一条静稳负对照。参考范围实验因为输入的确认活动为空，直接跳过模型；该次空输出不能作为模型主动识别静稳的证据。可切回当前实验查看真实视觉调用。",
  case_0023: "阶段判别与开发标签存在较多差异。尾部第 1020 日附近的 N 原始位移再次抬升，现有参考边界也需要独立复核；不能只依据标签就认定所有尾部预测都错。单次视觉配置还存在共享闭区间端点的问题，应与识别分歧分开。",
};
const state = { data: null, caseId: "case_0003", run: "visual-v3", method: "visual_model", step: "stage", tab: "prompt" };
const $ = (id) => document.getElementById(id);

function isActivityRequest(kind) {
  return kind === "locate" || kind === "locate-global" || kind === "locate-local";
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) {
    node.textContent = text;
  }
  if (className) {
    node.className = className;
  }
  return node;
}

function selectedRequest() {
  let run = state.run;
  let kind = state.method;
  if (state.step === "locate") {
    if (!localMethods.slice(0, 2).includes(state.method)) {
      run = ["visual-v3", "local-v1"].includes(state.run) ? state.data.localization_source : state.run;
      kind = "locate";
    }
  }
  return state.data.requests.find((row) => row.case_id === state.caseId && row.run === run && row.kind === kind);
}

function figure(asset, sent, index) {
  const frame = element("figure");
  const button = element("button", undefined, "image-button");
  button.setAttribute("aria-label", `放大 ${asset.name}`);
  const img = element("img");
  img.src = asset.url;
  img.alt = asset.name;
  img.loading = "lazy";
  button.append(img);
  button.onclick = () => {
    $("dialog-title").textContent = (sent ? "实际输入 / " : "事后审查 / ") + asset.name;
    $("dialog-image").src = asset.url;
    $("dialog-image").alt = img.alt;
    $("original-link").href = asset.url;
    $("image-dialog").showModal();
  };
  const caption = element("figcaption");
  const line = element("div", undefined, "image-title");
  let label = asset.name;
  if (sent) {
    let imageType = asset.name;
    if (asset.name.endsWith("-raw.png")) {
      imageType = "三轴原始位移 / 全局图";
    } else if (/-local-[123]\.png$/.test(asset.name)) {
      imageType = "局部原始位移 / 固定 180 日窗";
    } else if (asset.name.endsWith("-auxiliary.png")) {
      imageType = "位移＋速度＋加速度 / 61 日辅助图";
    }
    label = `${index + 1}. ${imageType}`;
  }
  line.append(element("span", label), element("span", sent ? "SHA256 已核对 ✓" : "未发送", sent ? "" : "tag amber"));
  caption.append(line, element("div", `${asset.name}\nSHA256 ${asset.sha256}`, "image-hash"));
  frame.append(button, caption);
  return frame;
}

function timeline(label, values, stage, dates, missing) {
  const row = element("div", undefined, "timeline-row");
  row.append(element("span", label, "timeline-label"));
  const track = element("div", undefined, "track");
  track.setAttribute("aria-label", label + "；逐日掩码，鼠标悬停查看区间");
  let start = 0;
  function segment(left, stop, css, description) {
    const bar = element("span", undefined, css);
    bar.style.left = `${left / values.length * 100}%`;
    bar.style.width = `${(stop - left) / values.length * 100}%`;
    bar.title = `${description}：第 ${left}–${stop - 1} 日 / ${dates[left]} 至 ${dates[stop - 1]}`;
    track.append(bar);
  }
  for (let end = 1; end <= values.length; end++) {
    if (end < values.length && values[end] === values[start]) {
      continue;
    }
    const value = values[start];
    if (value !== 0) {
      let css = "active";
      let description = "确认活动";
      if (value === -1) {
        css = "unknown";
        description = "未知";
      } else if (stage) {
        css = `stage${value}`;
        description = ["", "A 加速", "S 匀速", "D 减速"][value];
      }
      segment(start, end, css, description);
    }
    start = end;
  }
  for (const day of missing) {
    segment(day, day + 1, "missing", "缺测");
  }
  row.append(track);
  return row;
}

function renderTimeline(request) {
  const evidence = state.data.cases[state.caseId];
  const saved = request ? request.accepted : evidence.reference_prediction;
  const body = $("timelines");
  body.replaceChildren(timeline("已接收活动", saved.activity, false, evidence.dates, evidence.missing));
  if (!isActivityRequest(request?.kind)) {
    body.append(timeline("已接收阶段", saved.stage, true, evidence.dates, evidence.missing));
  }
  const axis = element("div", undefined, "axis");
  axis.append(element("span", `0 / ${evidence.dates[0]}`), element("span", "零基日索引 · 含端点"), element("span", `${evidence.dates.length - 1} / ${evidence.dates.at(-1)}`));
  body.append(axis);
  let note = "缺测保留原日历位置，灰色为未知，底部深灰细线为缺测；上图是解析后的已接收结果，未作为输入图片发送。";
  if (isActivityRequest(request?.kind)) {
    note += " 本步只定位活动，不判阶段。";
  }
  if (request?.validation.status === "failed") {
    note += isActivityRequest(request.kind)
      ? " 本次定位回答被拒绝：活动整项保持未知，未自动裁剪或修复。"
      : " 本次原始回答被拒绝：失败阶段保持未知，未自动裁剪或修复。";
  }
  if (request?.kind === "visual_model_ungated") {
    note += " 此消融允许在活动范围外输出阶段。";
  }
  $("timeline-note").textContent = note;
}

function metadata(pairs) {
  const list = element("dl", undefined, "metadata");
  for (const [label, value] of pairs) {
    const item = element("div");
    item.append(element("dt", label), element("dd", value ?? "未记录"));
    list.append(item);
  }
  return list;
}

function copyBlock(label, text) {
  const block = element("div");
  const toolbar = element("div", undefined, "toolbar");
  const button = element("button", "复制原文", "copy-button");
  const pre = element("pre", text);
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "已复制 ✓";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "已选中，请 Ctrl+C";
    }
  };
  toolbar.append(element("span", label), button);
  block.append(toolbar, pre);
  return block;
}

function renderReceipt(request) {
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === state.tab));
    button.tabIndex = button.dataset.tab === state.tab ? 0 : -1;
  }
  const body = $("receipt-body");
  body.setAttribute("aria-labelledby", `tab-${state.tab}`);
  body.replaceChildren();
  if (!request) {
    body.append(element("div", "本例参考活动范围为空，程序直接输出无阶段。没有调用模型，因此没有提示词、发送图片或模型回答。", "empty"));
    return;
  }
  const response = request.response;
  if (state.tab === "prompt") {
    const guide = isActivityRequest(request.kind)
      ? "中文导读：先从原始位移找持续累积的活动范围；平台、单次跳变和孤立导数峰不足以确认活动，不明确处返回 uncertain。本步不判断阶段。"
      : "中文导读：结合原始位移与可用辅助证据，判断速率增加 A、近似恒定的非零速率 S、速率降低 D。缓慢运动不自动等于 S，速度高不自动等于 A，过渡或证据不足时允许不标。具体门控或解除门控要求以原文为准。";
    body.append(element("p", guide, "notice"));
    body.append(element("p", "来源：本次 .started.json 中的完整英文原文。图片按左侧顺序放在同一条 user 消息中。", "small"));
    body.append(copyBlock("原始提示词 / 未翻译、未改写", request.prompt));
    body.append(metadata(Object.entries(request.parameters)));
    body.append(element("p", "enable_thinking=false；response_format=json_object。两项来自已核查的调用代码；用图片字节和提示词重建请求后，整体 SHA256 与日志一致。重建校验不代表保存了网络报文。", "small"));
  } else if (state.tab === "response") {
    body.append(metadata([
      ["请求返回状态", response.status], ["HTTP", response.http_status],
      ["实际返回模型", response.returned_model], ["耗时（秒）", response.seconds?.toFixed(3)],
      ["总 token", response.usage?.total_tokens], ["结束原因", response.finish_reason],
    ]));
    body.append(copyBlock("message.content / 模型原始字符串", response.output ?? "未保存模型输出"));
    if (response.output) {
      try {
        const formatted = element("details");
        formatted.append(element("summary", "格式化 JSON（仅改变缩进）"), element("pre", JSON.stringify(JSON.parse(response.output), null, 2)));
        body.append(formatted);
      } catch {
        body.append(element("p", "输出无法解析为 JSON。", "notice error"));
      }
    }
    body.append(element("p", "模型只返回区间 JSON。上面的原文没有解释其判断原因，本页不补造模型判读过程。", "notice"));
  } else {
    const check = request.validation;
    body.append(element("div", check.status === "passed" ? "当前解析器重放通过，与保存预测一致。这里只确认格式、区间与门控契约。" : "原始回答未通过契约检查；原实验已按失败处理，未自动修正。", check.status === "passed" ? "notice" : "notice error"));
    const list = element("ul", undefined, "check-list");
    const checks = ["发送图片 SHA256 全部一致", "完整请求重建 SHA256 一致", "开始日志与响应日志中的输入字段一致",
      "重放成功 / 失败状态与冻结记录一致", "缺测掩码来自已核对哈希的原始 input.json"];
    for (const message of checks) {
      list.append(element("li", message));
    }
    body.append(list);
    if (check.error) {
      body.append(copyBlock("程序拒绝原因 / 非模型解释", check.error));
    }
    for (const detail of check.details) {
      body.append(element("p", detail.message, "notice error"));
    }
    if (!check.details.length) {
      const contractNote = isActivityRequest(request.kind)
        ? "未检测到活动与未知区间重叠；本步不判阶段。"
        : "没有检测到阶段重叠或有观测日超出门控。";
      body.append(element("p", contractNote + "识别是否合理仍需结合原始位移和下方开发标注审查。", "small"));
    }
    const gate = element("details");
    const spans = [];
    let start = null;
    request.activity.forEach((value, index) => {
      if (value === 1 && start === null) {
        start = index;
      }
      if (start !== null && (value !== 1 || index === request.activity.length - 1)) {
        spans.push([start, value === 1 ? index : index - 1]);
        start = null;
      }
    });
    gate.append(element("summary", "查看本步活动范围（应用缺测后）"), element("pre", JSON.stringify(spans, null, 2)));
    body.append(gate);
  }
  const links = element("div", undefined, "download-links");
  for (const [key, url] of Object.entries(request.logs)) {
    const link = element("a", key === "started" ? "下载请求日志 ↗" : "下载响应日志 ↗");
    link.href = url;
    link.download = "";
    links.append(link);
  }
  body.append(links, element("p", `请求 ID：${request.id}`, "small"));
}

function renderAudit() {
  const box = $("audit");
  box.replaceChildren();
  const localRun = state.run === "local-v1";
  const notePrefix = localRun ? "既有复核背景（本轮实验前，非本轮模型解释）：" : "";
  box.append(element("p", notePrefix + reviewNotes[state.caseId], "notice"));
  const source = element("a", "既有诊断说明（程序统计与原图复核，非模型解释） ↗");
  source.href = "diagnosis-notes.md";
  source.target = "_blank";
  source.rel = "noopener";
  box.append(source);
  const method = state.method === "single" ? "single_visual" : state.method;
  const audit = localRun ? state.data.local_experiment.cases[state.caseId].audit[method] : state.data.cases[state.caseId].audit[method];
  if (!["visual-v3", "local-v1"].includes(state.run) || !audit) {
    box.append(element("p", "此处误差计数仅对当前 visual-v3 端到端实验提供。参考活动范围来自开发标签，不作为可部署定位结果评分；历史轮次请看其原始输出与契约记录。", "notice"));
  } else {
    const sourceName = localRun ? "本轮 error-audit.json" : "existing-error-audit.json";
    box.append(element("p", `来源：${sourceName} / 当前配置与 AI 辅助开发标注的差异（逐日计数）`, "small"));
    const cards = element("div", undefined, "audit-cards");
    const wrongStageDays = isActivityRequest(state.method) ? "—" : audit.stages.counts.stage_wrong_class;
    for (const [label, value] of [["活动误报日", audit.activity.false_activity.days], ["活动漏检日", audit.activity.missed_activity.days], ["范围内阶段错分日", wrongStageDays], ["弱位移漏检日", audit.activity.weak_missed.days]]) {
      const card = element("div", undefined, "audit-card");
      card.append(element("b", value), element("span", label));
      cards.append(card);
    }
    box.append(cards);
    const details = element("details");
    details.append(element("summary", "完整错误分区与混淆矩阵（来源区间为半开区间）"), element("pre", JSON.stringify(audit, null, 2)));
    box.append(details);
  }
  $("review-images").replaceChildren();
  $("review-gallery").open = false;
  $("review-gallery-title").textContent = localRun
    ? "展开本轮活动与阶段对比结果图"
    : "展开阶段对比图与全部局部原始图";
  $("review-note").textContent = localRun
    ? "以下误差对照 AI 辅助开发标注。下方对比结果图未发送给模型；当前请求实际使用的图片完整显示在上方输入区，局部配置确实发送了三页局部原图。定位与阶段实验独立，不把新定位拼成阶段实验的门控。"
    : "以下误报、漏检和阶段差异对照 AI 辅助开发标注；它们提示复核方向，不等于已证实的物理错误。下方局部图供事后审查，本轮历史请求没有看到这些局部窗口。";
}

function renderFlow() {
  const localActivity = localMethods.slice(0, 2).includes(state.method);
  const visualGate = ["visual_model", "visual_model_raw", "visual_model_ungated", "stage-global", "stage-local"].includes(state.method);
  let firstLabel;
  let firstNote;
  if (localActivity) {
    firstLabel = "确认活动范围";
    firstNote = state.method === "locate-local" ? "全局＋三页固定局部原图" : "全局原图 · 同期控制";
  } else if (visualGate) {
    firstLabel = "确认活动范围";
    firstNote = ["visual-v3", "local-v1"].includes(state.run) ? "复用 visual-v2 · 点击查看原调用" : "原始位移图 → 视觉定位";
  } else if (state.method === "numeric_model") {
    firstLabel = "数值确认活动范围";
    firstNote = "鲁棒数值规则 · 无视觉定位调用";
  } else if (state.method === "single") {
    firstLabel = "单次联合输出";
    firstNote = "原图＋辅助图 → 活动与阶段";
  } else {
    firstLabel = "提供参考活动范围";
    firstNote = "AI 开发标签 · 仅用于诊断";
  }
  const buttons = [];
  const skipped = state.run === "reference-v1" && state.data.reference_skipped.includes(state.caseId);
  let stageNote = state.method === "visual_model_raw" ? "仅原图 · 去除辅助证据" : "原图＋辅助图 · A / S / D";
  if (skipped) {
    stageNote = "确认活动为空 · 无模型调用";
  }
  if (localActivity) {
    stageNote = "定位单独实验 · 不输出阶段";
  } else if (state.method === "stage-local") {
    stageNote = "固定旧门控 · 原图＋辅助图＋局部图";
  }
  const steps = [
    ["01", firstLabel, firstNote, "locate"],
    ["02", state.method === "single" ? "查看联合回答" : "判别活动内部阶段", stageNote, "stage"],
    ["03", "程序校验与审查", "严格边界 / 缺测 / 失败保留", "validation"],
  ];
  for (const [index, title, note, step] of steps) {
    const button = element("button");
    const active = step === "validation" ? state.tab === "validation" : (state.step === step && state.tab !== "validation");
    button.className = active ? "selected" : "";
    button.setAttribute("aria-pressed", String(active));
    const text = element("span", title);
    text.append(element("small", note));
    button.append(element("b", index), text);
    button.disabled = (step === "locate" && !visualGate && !localActivity) || (step === "stage" && localActivity);
    button.onclick = () => {
      if (step === "validation") {
        state.tab = "validation";
      } else {
        state.step = step;
        state.tab = "prompt";
      }
      render();
    };
    buttons.push(button);
  }
  $("flow").replaceChildren(...buttons);
}

function renderFailures() {
  const links = $("failure-links");
  links.replaceChildren();
  if (!$("failed-only").checked) {
    return;
  }
  const failed = state.data.requests.filter((row) => row.run === state.run && row.validation.status === "failed");
  if (!failed.length) {
    links.append(element("p", "本轮没有契约失败。", "small"));
  }
  for (const row of failed) {
    const button = element("button", `${row.case_id} · ${methods[row.kind] ?? row.kind}`);
    button.onclick = () => {
      state.caseId = row.case_id;
      state.method = row.kind;
      state.step = isActivityRequest(row.kind) ? "locate" : "stage";
      state.tab = "validation";
      render();
    };
    links.append(button);
  }
}

function render() {
  const request = selectedRequest();
  for (const button of document.querySelectorAll(".case-button")) {
    button.setAttribute("aria-current", String(button.dataset.case === state.caseId));
  }
  $("case-title").textContent = `${state.caseId} / ${caseNotes[state.caseId]}`;
  $("case-meta").textContent = `${state.run} · ${methods[state.method]} · 1095 DAYS`;
  $("method").value = state.method;
  const passed = request?.validation.status === "passed";
  let status = "未调用模型";
  if (request) {
    status = passed ? "输入已核验 · 契约通过" : "输入已核验 · 契约失败";
  }
  $("status").replaceChildren(element("span", status, "tag" + (request && !passed ? " fail" : "")));
  let source = "";
  if (state.run === "local-v1") {
    source = isActivityRequest(state.method)
      ? "本轮定位受控对照：同一提示词与全局原图，局部组增加全部固定 180 日窗口。只有原始位移图，不提供导数或标签；该输出不用于独立阶段对照的门控。"
      : "本轮阶段受控对照：门控直接复用 visual-v3 的旧视觉活动，完整提示词与原图/辅助图不变；局部组增加三页固定原始局部图。步骤 01 可追溯到 visual-v2 定位请求。";
  } else if (state.run === "reference-v1") {
    source = "参考范围诊断：活动范围来自 AI 辅助开发标签，未提供阶段标签。该配置不属于可部署方法；case_0015 无活动，程序跳过模型调用。";
  } else if (state.method === "numeric_model") {
    source = "第一步来自冻结的鲁棒数值规则；第二步使用当前请求原图与辅助图。提示中的活动列表已与保存的数值门控逐日核对。";
  } else if (state.method === "single") {
    source = "本项只有一次模型调用，同时输出活动区间和阶段；不会使用另一次视觉定位结果。任一整体契约失败时整项预测保持未知。";
  } else {
    source = state.run === "visual-v3" ? "定位来源：visual-v2 / locate → 缺测掩码 → 复用到 visual-v3 阶段请求。点击步骤 01 可查看当时真实提示词、原图与回答；并非本轮重新定位。" : "历史轮次：第一步只看原始位移图；第二步接收确认范围并判阶段。此轮保留了相邻闭区间共享端点等失败，未修补原输出。";
  }
  if (state.method === "visual_model_ungated") {
    source += " 本消融解除活动约束，活动范围仅作上下文，允许在范围外输出阶段。";
  }
  $("source-note").textContent = source;
  renderFlow();
  renderTimeline(request);
  $("input-images").replaceChildren();
  $("image-count").textContent = `${request?.images.length ?? 0} 张 · 按请求顺序`;
  if (request) {
    request.images.forEach((asset, index) => $("input-images").append(figure(asset, true, index)));
  } else {
    $("input-images").append(element("div", "该记录没有请求，不展示其他调用的图片作为本次输入。", "empty"));
  }
  renderReceipt(request);
  renderAudit();
  renderFailures();
}

function configureMethods() {
  const select = $("method");
  select.replaceChildren();
  for (const [key, label] of Object.entries(methods)) {
    if ((state.run === "local-v1") !== localMethods.includes(key)) {
      continue;
    }
    if ((state.run === "reference-v1") !== (key === "reference-stage")) {
      continue;
    }
    const option = element("option", label);
    option.value = key;
    select.append(option);
  }
  if (![...select.options].some((option) => option.value === state.method)) {
    state.method = select.options[0].value;
  }
  state.step = isActivityRequest(state.method) ? "locate" : "stage";
  select.value = state.method;
}

async function initialize() {
  const response = await fetch("trace-data.json");
  if (!response.ok) {
    throw new Error(`审查数据读取失败：HTTP ${response.status}`);
  }
  state.data = await response.json();
  if (state.data.local_experiment) {
    const option = element("option", "局部图实验 · local-v1");
    option.value = "local-v1";
    $("run").append(option);
    state.run = "local-v1";
    state.method = "stage-local";
    $("run").value = state.run;
    $("trace-footer").textContent = "本次局部图实验新增 24 次模型调用，保留全部原始回答与失败；历史实验与标签未改。通过契约不代表判断正确。固定视觉范围＋XGBoost 的数值推理结果见上一级实验结果页。居中导数仅支持离线回顾。";
  }
  const summary = state.data.summary;
  $("overview").replaceChildren();
  for (const [value, label] of [[Object.keys(state.data.cases).length, "条开发记录"], [summary.requests, "次保存的实际调用"], [summary.images_checked, "次图片哈希核验"], [summary.contract_failures, "次契约失败可追溯"], [summary.new_model_calls, "次新增调用"]]) {
    const metric = element("div", undefined, "metric");
    metric.append(element("b", value), element("span", label));
    $("overview").append(metric);
  }
  for (const cid of Object.keys(state.data.cases)) {
    const button = element("button", undefined, "case-button");
    button.dataset.case = cid;
    const text = element("span");
    text.append(element("strong", cid), element("small", caseNotes[cid]));
    button.append(text, element("span", undefined, "case-dot"));
    button.onclick = () => {
      state.caseId = cid;
      render();
    };
    $("case-list").append(button);
  }
  configureMethods();
  $("run").onchange = (event) => {
    state.run = event.target.value;
    state.step = "stage";
    configureMethods();
    render();
  };
  $("method").onchange = (event) => {
    state.method = event.target.value;
    state.step = isActivityRequest(state.method) ? "locate" : "stage";
    render();
  };
  $("failed-only").onchange = renderFailures;
  const tabs = [...document.querySelectorAll("[data-tab]")];
  for (const button of tabs) {
    button.onclick = () => {
      state.tab = button.dataset.tab;
      renderReceipt(selectedRequest());
      renderFlow();
    };
    button.onkeydown = (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        return;
      }
      event.preventDefault();
      let index = tabs.indexOf(button);
      if (event.key === "Home") {
        index = 0;
      } else if (event.key === "End") {
        index = tabs.length - 1;
      } else {
        index = (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      }
      tabs[index].click();
      tabs[index].focus();
    };
  }
  $("review-gallery").ontoggle = () => {
    if ($("review-gallery").open && !$("review-images").childElementCount) {
      const evidence = state.run === "local-v1" ? state.data.local_experiment.cases[state.caseId] : state.data.cases[state.caseId];
      for (const asset of evidence.review_images) {
        $("review-images").append(figure(asset, false));
      }
    }
  };
  $("close-dialog").onclick = () => $("image-dialog").close();
  render();
}

initialize().catch((error) => {
  $("overview").textContent = `无法加载审查记录：${error.message}。请通过本地 HTTP 服务打开此页。`;
});
