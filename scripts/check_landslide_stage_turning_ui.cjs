/* Script behavior checks in a DOM stub; not real browser verification. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const page = path.resolve(process.argv[2]);
const html = fs.readFileSync(page, "utf8");
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payload);
const nodes = {};
function node(id) {
  return {
    value: id === "view-select" ? "0" : id === "arm-select" ? "image-r1" : "",
    textContent: id === "page-data" ? payload : "", hidden: false, events: {},
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
let requestStates = 0;
let skippedStates = 0;
for (let caseIndex = 0; caseIndex < data.cases.length; caseIndex++) {
  const c = data.cases[caseIndex];
  nodes["case-select"].value = String(caseIndex);
  for (let view = 0; view < 4; view++) {
    nodes["view-select"].value = String(view);
    for (const arm of ["image-r1", "numeric-r1", "image-r2", "numeric-r2"]) {
      nodes["arm-select"].value = arm;
      nodes["arm-select"].events.change();
      assert.equal(nodes["case-title"].textContent, c.case_id);
      assert.equal(nodes.comparison.src, c.plots[view]);
      assert(fs.existsSync(path.join(path.dirname(page), c.plots[view])));
      assert.deepEqual(JSON.parse(nodes.candidates.textContent), c.summary);
      const request = c.requests[arm];
      assert.equal(nodes["request-detail"].hidden, !request);
      if (request) {
        assert.equal(nodes.prompt.textContent, request.started.prompt);
        assert.equal(nodes.response.textContent, request.response.output || "没有完整回答，已保留弃判");
        assert.equal([...nodes.inputs.innerHTML.matchAll(/<img /g)].length, 7);
        for (const image of request.images) {
          assert(fs.existsSync(path.join(path.dirname(page), image)));
        }
        requestStates++;
      } else {
        assert.equal(nodes.prompt.textContent, "");
        assert.equal(nodes.inputs.innerHTML, "");
        assert(nodes["request-status"].textContent.includes("不计正常识别成绩"));
        skippedStates++;
      }
      states++;
    }
  }
}
assert.equal(states, 192);
assert.equal(requestStates, 144);
assert.equal(skippedStates, 48);
console.log(JSON.stringify({ status: "passed", states, actual_request_states: requestStates,
  skipped_states: skippedStates, actual_prompts_outputs_and_seven_images_match: true,
  limitation: "Node DOM stub; browser rendering and real interaction remain unverified" }, null, 2));
