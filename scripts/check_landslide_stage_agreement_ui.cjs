/* DOM script checks only; no browser rendering certification. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const pagePath = path.resolve(process.argv[2]);
const html = fs.readFileSync(pagePath, "utf8");
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payload);
const defaults = { "mode-select": "final", "view-select": "0", "window-select": "1" };
const nodes = {};
function node(id) {
  return {
    value: defaults[id] || "", textContent: id === "page-data" ? payload : "",
    events: {}, attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.events[name] = callback; },
    set innerHTML(value) {
      this.html = value;
      if (id === "case-select") { this.value = value.match(/value="([^"]+)"/)[1]; }
    },
    get innerHTML() { return this.html || ""; },
  };
}
const context = vm.createContext({ document: {
  getElementById(id) {
    if (!nodes[id]) { nodes[id] = node(id); }
    return nodes[id];
  },
} });
function evaluate(code) { return vm.runInContext(code, context); }
evaluate(script);
assert.equal(data.cases.length, 12);
assert.equal(data.summary.methods.final.visual.reference_evaluable_days, 585);
const colors = { acceleration: "#d27545", steady_motion: "#557ab6", deceleration: "#a98630", unknown: "#e5b65b", none: "#dbe7d4" };
let checkedColors = 0;
let states = 0;
let noReferenceCases = 0;
for (let caseIndex = 0; caseIndex < data.cases.length; caseIndex++) {
  const c = data.cases[caseIndex];
  nodes["case-select"].value = String(caseIndex);
  if (!c.scores.final.rule.reference_evaluable_days) { noReferenceCases++; }
  for (const mode of ["native", "final"]) {
    nodes["mode-select"].value = mode;
    for (let view = 0; view < 4; view++) {
      nodes["view-select"].value = String(view);
      nodes["view-select"].events.change();
      assert.equal(nodes["case-title"].textContent, c.case_id);
      assert.equal(nodes["raw-image"].src, c.raw_images[view]);
      assert(fs.existsSync(path.join(path.dirname(pagePath), nodes["raw-image"].src)));
      assert(nodes["totals"].innerHTML.includes(mode === "final" ? "414/585" : "439/585"));
      const ranges = [[0, c.observed.length], [0, 410], [320, 775], [685, c.observed.length]];
      const [start, stop] = ranges[view];
      const values = [c.reference_features, ...["rule", "xgboost", "visual"].map(method => c.prediction_features[mode][method])];
      const preview = [...nodes.bands.innerHTML.matchAll(/<rect[^>]*fill="([^"]+)"/g)].map(match => match[1]);
      assert.equal(preview.length, 4 * (stop - start));
      for (let row = 0; row < values.length; row++) {
        for (let day = start; day < stop; day++) {
          const expected = c.observed[day] ? colors[values[row][day]] : "#737373";
          assert.equal(preview[row * (stop - start) + day - start], expected);
          checkedColors++;
        }
      }
      if (!c.interiors.length) {
        assert(nodes.interiors.innerHTML.includes("不计正常识别成绩"));
      }
      states++;
    }
  }
  for (let window = 0; window < 3; window++) {
    nodes["window-select"].value = String(window);
    nodes["window-select"].events.change();
    assert.equal(nodes["aux-image"].src, c.auxiliary_images[window]);
    assert(fs.existsSync(path.join(path.dirname(pagePath), nodes["aux-image"].src)));
  }
}
console.log(JSON.stringify({
  status: "passed", states_checked: states, timeline_colors_checked: checkedColors,
  auxiliary_selections_checked: 36, zero_reference_cases: noReferenceCases,
  label_calendar_and_missing_colors_match: true,
  mode_table_matches: true, no_activity_not_normal_score: true,
  limitation: "Node DOM stub only; rendering, accessibility and real browser interactions unverified",
}, null, 2));
