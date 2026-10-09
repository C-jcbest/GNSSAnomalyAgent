/* Supplemental page behavior checks; a DOM stub is not browser acceptance. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const page = path.resolve(process.argv[2]);
const html = fs.readFileSync(page, "utf8");
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const base = html.match(/<base href="([^"]+)">/)[1];
const assets = path.resolve(path.dirname(page), base);
const data = JSON.parse(payload);
const nodes = {};
function node(id) {
  return {
    value: id === "view-select" ? "0" : id === "arm-select" ? "image-r1" : "",
    textContent: id === "page-data" ? payload : "", events: {},
    addEventListener(name, callback) { this.events[name] = callback; },
    set innerHTML(value) {
      this.html = value;
      if (id === "case-select") { this.value = value.match(/value="([^"]+)"/)[1]; }
    },
    get innerHTML() { return this.html || ""; },
  };
}
const context = vm.createContext({ document: { getElementById(id) {
  if (!nodes[id]) { nodes[id] = node(id); }
  return nodes[id];
} } });
vm.runInContext(script, context);
let states = 0;
let rejectedStates = 0;
for (let index = 0; index < data.cases.length; index++) {
  const c = data.cases[index];
  nodes["case-select"].value = String(index);
  for (let view = 0; view < 4; view++) {
    nodes["view-select"].value = String(view);
    for (const arm of ["image-r1", "numeric-r1", "image-r2", "numeric-r2"]) {
      nodes["arm-select"].value = arm;
      nodes["arm-select"].events.change();
      assert.equal(nodes.comparison.src, c.plots[view]);
      assert(fs.existsSync(path.join(assets, c.plots[view])));
      assert(nodes["record-diagnosis"].textContent.length > 0);
      const request = c.requests[arm];
      if (request) {
        assert.equal(nodes.prompt.textContent, request.started.prompt);
        assert.equal(nodes.response.textContent, request.response.output);
        const status = data.summary.results.find(r => r.request_id === `${c.case_id}-${arm}`);
        assert(nodes["request-status"].textContent.includes(status.status));
        assert.equal(JSON.parse(nodes.receipt.textContent).stage_contract_status, status.status);
        if (status.error) {
          assert(nodes["request-status"].textContent.includes(status.error.message));
          assert(nodes["request-status"].textContent.includes("整记录阶段未知，无重试"));
          rejectedStates++;
        }
        assert.equal([...nodes.inputs.innerHTML.matchAll(/<img /g)].length, 7);
        for (const image of request.images) { assert(fs.existsSync(path.join(assets, image))); }
      } else {
        assert.equal(nodes.prompt.textContent, "");
        assert.equal(nodes.inputs.innerHTML, "");
        assert(nodes["request-status"].textContent.includes("不计正常识别成绩"));
      }
      states++;
    }
  }
}
assert.equal(states, 192);
assert.equal(rejectedStates, 4);
console.log(JSON.stringify({ status: "passed", states, explicit_rejected_states: rejectedStates,
  actual_prompts_outputs_and_seven_images_match: true, base_path_assets_exist: true,
  limitation: "Node DOM stub; browser rendering remains unverified" }, null, 2));
