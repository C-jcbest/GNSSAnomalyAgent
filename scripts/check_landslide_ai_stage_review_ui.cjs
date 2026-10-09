/* Script behavior only: this DOM stub does not certify browser rendering. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const pagePath = path.resolve(process.argv[2]);
const dailyPath = path.resolve(process.argv[3]);
const html = fs.readFileSync(pagePath, "utf8");
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const base = html.match(/<base href="([^"]+)"/);
const assetDirectory = base ? path.resolve(path.dirname(pagePath), base[1]) : path.dirname(pagePath);
const input = JSON.parse(payload);
const daily = JSON.parse(fs.readFileSync(dailyPath, "utf8"));
const rowMap = new Map(daily.map(row => [`${row.case_id}:${row.day_index}`, row]));
const nodes = {};
const defaults = { "view-select": "0", "window-select": "1" };
const memory = new Map();
const downloads = [];
const blobs = [];

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

const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!nodes[id]) { nodes[id] = element(id); }
      return nodes[id];
    },
    createElement() { return element("created-anchor"); },
  },
  Blob,
  URL: {
    createObjectURL(blob) { blobs.push(blob); return "blob:script-check"; },
    revokeObjectURL() {},
  },
  localStorage: {
    getItem(key) { return memory.get(key) || null; },
    setItem(key, value) { memory.set(key, value); },
  },
});
vm.runInContext(script, context);
function evaluate(code) { return vm.runInContext(code, context); }
function imageExists(name) { return fs.existsSync(path.join(assetDirectory, name)); }

async function main() {
  assert.equal(input.cases.length, 12);
  assert.equal(evaluate("completeErrors().length"), 0, "Completed AI packet was rejected");
  assert.deepEqual(JSON.parse(evaluate("JSON.stringify(packet)")), input.template);
  const expectedColors = {
    acceleration: "#d27545", steady_motion: "#557ab6", deceleration: "#a98630", unknown: "#e5b65b",
  };
  const counts = { acceleration: 0, steady_motion: 0, deceleration: 0, unknown: 0, missing: 0 };
  let interiors = 0;
  let noActivity = 0;
  let rawViews = 0;
  let auxiliaryViews = 0;
  let unknownEvidenceFields = 0;
  for (let caseIndex = 0; caseIndex < input.cases.length; caseIndex++) {
    nodes["case-select"].value = String(caseIndex);
    evaluate("chooseCase()");
    const sourceCase = input.template.cases[caseIndex];
    if (!sourceCase.interiors.length) {
      noActivity++;
      assert(nodes["stage-form"].hidden);
      assert(nodes["aux-content"].hidden);
    }
    for (let view = 0; view < 4; view++) {
      nodes["view-select"].value = String(view);
      evaluate("renderImages()");
      assert.equal(nodes["raw-image"].src, input.cases[caseIndex].raw_images[view]);
      assert(imageExists(nodes["raw-image"].src));
      rawViews++;
    }
    for (let interiorIndex = 0; interiorIndex < sourceCase.interiors.length; interiorIndex++) {
      interiors++;
      nodes["interior-select"].value = String(interiorIndex);
      evaluate("renderSegments()");
      assert(!nodes["aux-content"].hidden);
      const interior = sourceCase.interiors[interiorIndex];
      if (base) {
        const rateFields = [...nodes["segments-list"].innerHTML.matchAll(/data-field="rate_evidence"/g)];
        assert.equal(rateFields.length, interior.segments.length, "Unknown rate evidence stayed hidden");
        unknownEvidenceFields += interior.segments.filter(s => s.feature === "unknown").length;
      }
      const colors = [...nodes["stage-preview"].innerHTML.matchAll(/fill="([^"]+)"/g)].map(match => match[1]);
      assert.equal(colors.length, interior.stop - interior.start);
      for (let day = interior.start; day < interior.stop; day++) {
        const row = rowMap.get(`${sourceCase.case_id}:${day}`);
        assert(row, "Preview date missing from exported rows");
        let expected;
        if (!input.cases[caseIndex].observed[day]) {
          expected = "#737373";
          counts.missing++;
          assert.equal(row.stage_evaluable, false);
          assert.notEqual(row.activity_label, 1);
        } else {
          assert.equal(row.activity_label, 1);
          expected = expectedColors[row.feature];
          assert(expected, "Exported active feature has no preview color");
          counts[row.feature]++;
          assert.equal(row.stage_evaluable, row.feature !== "unknown");
        }
        assert.equal(colors[day - interior.start], expected, `Preview/export mismatch ${sourceCase.case_id}:${day}`);
      }
      for (let window = 0; window < 3; window++) {
        nodes["window-select"].value = String(window);
        evaluate("renderImages()");
        assert.equal(nodes["aux-image"].src, input.cases[caseIndex].auxiliary_images[window]);
        assert(imageExists(nodes["aux-image"].src));
        auxiliaryViews++;
      }
    }
  }
  assert.equal(interiors, 21);
  assert.equal(noActivity, 3);
  assert.deepEqual(counts, { acceleration: 290, steady_motion: 25, deceleration: 270, unknown: 1541, missing: 84 });
  nodes["export-complete"].events.click();
  assert.equal(downloads[0], "stage-review-completed.json");
  assert.deepEqual(JSON.parse(await blobs[0].text()), input.template);
  nodes["case-select"].value = "0";
  evaluate("chooseCase()");
  nodes["segments-list"].events.change({ target: { dataset: { field: "raw_evidence", index: "0" }, value: "SCRIPT CHECK ONLY" } });
  assert.equal(evaluate("current().interior.review_status"), "unreviewed");
  assert(evaluate("completeErrors().length") > 0);
  nodes["export-complete"].events.click();
  assert.equal(downloads.length, 1, "Edited review was exported as complete");
  console.log(JSON.stringify({
    status: "passed", page: pagePath, cases: 12, interiors, no_activity_cases: noActivity,
    counts, raw_view_combinations: rawViews, auxiliary_view_combinations: auxiliaryViews,
    unknown_evidence_fields: unknownEvidenceFields, complete_download_matches_packet: true,
    edit_invalidates_completion: true,
    limitation: "Node DOM stub; not browser rendering, accessibility, file picker or storage persistence verification",
  }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
