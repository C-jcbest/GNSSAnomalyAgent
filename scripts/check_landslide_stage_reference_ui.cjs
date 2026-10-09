/* Frontend script checks in a DOM stub; does not certify browser rendering. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const pagePath = path.resolve(process.argv[2]);
const html = fs.readFileSync(pagePath, "utf8");
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const nodes = {};
const defaults = { "view-select": "0", "window-select": "1" };
const memory = new Map();
const downloads = [];

function element(id) {
  return {
    value: defaults[id] || "",
    textContent: id === "page-data" ? payload : "",
    hidden: false,
    checked: false,
    disabled: false,
    events: {},
    attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, handler) { this.events[name] = handler; },
    click() { downloads.push(this.download); },
    set innerHTML(value) {
      this.html = value;
      if (id.endsWith("-select")) {
        const first = value.match(/<option value="([^"]*)"/);
        this.value = first ? first[1] : "";
      }
    },
    get innerHTML() { return this.html || ""; },
  };
}

const document = {
  getElementById(id) {
    if (!nodes[id]) {
      nodes[id] = element(id);
    }
    return nodes[id];
  },
  createElement() { return element("created-anchor"); },
};
const context = vm.createContext({
  document, Blob,
  URL: { createObjectURL() { return "blob:script-test"; }, revokeObjectURL() {} },
  localStorage: {
    getItem(key) { return memory.get(key) || null; },
    setItem(key, value) { memory.set(key, value); },
  },
});
vm.runInContext(script, context);
const input = JSON.parse(payload);
function evaluate(code) { return vm.runInContext(code, context); }
function assert(condition, description) {
  if (!condition) {
    throw Error(description);
  }
}
function imageExists(name) {
  return fs.existsSync(path.join(path.dirname(pagePath), name));
}
assert(evaluate("completeErrors().length") > 0, "Empty template was treated as complete");
let rawViews = 0;
let auxiliaryViews = 0;
let noActivityCases = 0;
let missingPreviewDays = 0;
for (let caseIndex = 0; caseIndex < input.cases.length; caseIndex++) {
  nodes["case-select"].value = String(caseIndex);
  evaluate("chooseCase()");
  const sourceCase = input.template.cases[caseIndex];
  if (!sourceCase.interiors.length) {
    noActivityCases++;
    assert(nodes["stage-form"].hidden, "No activity retained a stale stage form");
    assert(nodes["aux-content"].hidden, "No activity unlocked auxiliary evidence");
  }
  for (let view = 0; view < 4; view++) {
    nodes["view-select"].value = String(view);
    evaluate("renderImages()");
    assert(nodes["raw-image"].src === input.cases[caseIndex].raw_images[view], "Wrong raw record/view");
    assert(imageExists(nodes["raw-image"].src), "Missing raw image");
    rawViews++;
  }
  for (let interiorIndex = 0; interiorIndex < sourceCase.interiors.length; interiorIndex++) {
    nodes["interior-select"].value = String(interiorIndex);
    evaluate("renderSegments()");
    assert(nodes["aux-content"].hidden, "Unreviewed raw data unlocked auxiliary views");
    const interior = sourceCase.interiors[interiorIndex];
    evaluate(`current().interior.segments=[{start:${interior.start},stop:${interior.stop},feature:'steady_motion',raw_evidence:'SCRIPT FIXTURE',rate_evidence:'SCRIPT FIXTURE',nonzero_evidence:'',steady_evidence:'',unknown_reason:null}];drawPreview()`);
    assert(evaluate("interiorErrors(current().interior).some(e=>e.includes('S需'))"), "S without nonzero/steady evidence was accepted");
    const colors = [...nodes["stage-preview"].innerHTML.matchAll(/fill="([^"]+)"/g)].map(m => m[1]);
    assert(colors.length === interior.stop - interior.start, "Preview calendar compressed");
    for (let day = interior.start; day < interior.stop; day++) {
      if (!input.cases[caseIndex].observed[day]) {
        assert(colors[day - interior.start] === "#737373", "Missing observation has definite color");
        missingPreviewDays++;
      }
    }
    evaluate("current().interior.raw_views_seen=true;current().interior.raw_activity_status='confirmed';current().interior.raw_activity_evidence='SCRIPT FIXTURE; NOT actual author/reference evidence';current().interior.segments=[];renderImages()");
    assert(!nodes["aux-content"].hidden, "Acknowledged raw review did not unlock auxiliary views");
    for (let window = 0; window < 3; window++) {
      nodes["window-select"].value = String(window);
      evaluate("renderImages()");
      assert(nodes["aux-image"].src === input.cases[caseIndex].auxiliary_images[window], "Wrong auxiliary record/window");
      assert(imageExists(nodes["aux-image"].src), "Missing auxiliary image");
      auxiliaryViews++;
    }
  }
}
evaluate("packet=clone(input.template)");
nodes["case-select"].value = "0";
evaluate("chooseCase()");
assert(evaluate("completeErrors().length") > 0, "Source reset became complete");
nodes["export-complete"].events.click();
assert(downloads.length === 0, "Incomplete review triggered complete download");
nodes["export-draft"].events.click();
assert(downloads[0] === "stage-review-draft.json", "Draft download path failed");
evaluate("validateImported(clone(input.template))");
assert(evaluate("(()=>{const x=clone(input.template);x.source_manifest_sha256='WRONG';try{validateImported(x);return false;}catch{return true;}})()"), "Changed source imported");
assert(evaluate("(()=>{const x=clone(input.template);x.cases.pop();try{validateImported(x);return false;}catch{return true;}})()"), "Dropped case imported");
assert(evaluate("(()=>{const x=clone(input.template);x.reviewer.prior_predictions_seen='false';try{validateImported(x);return false;}catch{return true;}})()"), "String exposure imported as boolean");
nodes["add-segment"].events.click();
assert(evaluate("current().interior.segments.length") === 1, "Add segment action failed");
evaluate("current().interior.review_status='reviewed';current().interior.reviewed_at='2026-10-08T00:00:00Z'");
nodes["segments-list"].events.change({ target: { dataset: { field: "raw_evidence", index: "0" }, value: "SCRIPT FIXTURE" } });
assert(evaluate("current().interior.review_status") === "unreviewed", "Edit did not invalidate completion");
const result = {
  status: "passed", raw_views: rawViews, auxiliary_views: auxiliaryViews,
  no_activity_cases: noActivityCases, missing_preview_days: missingPreviewDays,
  template_rejected_as_complete: true, draft_download_checked: true,
  source_and_exposure_import_guards: true, edit_invalidates_completion: true,
  research_reference_created: false,
  limitation: "Node DOM stub; not a browser rendering, accessibility, file-picker or persisted-storage test",
};
console.log(JSON.stringify(result, null, 2));
