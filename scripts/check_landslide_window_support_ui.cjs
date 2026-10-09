/* Real saved data in a DOM stub; does not verify browser rendering. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const page = path.resolve(process.argv[2]);
const html = fs.readFileSync(page,'utf8');
const payload = html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const audit = JSON.parse(payload);
const nodes = {};
const context = vm.createContext({document:{getElementById(id){
  if (!nodes[id]) {
    nodes[id] = {value:'0',textContent:id==='page-data'?payload:'',events:{},
      addEventListener(name,callback){this.events[name]=callback;}};
  }
  return nodes[id];
}}});
vm.runInContext(script,context);
let states = 0;
let images = 0;
let interiors = 0;
let empty = 0;
for (let index=0;index<audit.cases.length;index++) {
  const current = audit.cases[index];
  nodes['case-select'].value=String(index);
  nodes['case-select'].events.change();
  assert.equal(nodes['case-title'].textContent,current.case_id);
  assert.equal(nodes['audit-image'].src,current.plot);
  assert(fs.existsSync(path.join(path.dirname(page),current.plot)));
  assert.equal([...nodes['raw-images'].innerHTML.matchAll(/<img /g)].length,4);
  assert.equal([...nodes['aux-images'].innerHTML.matchAll(/<img /g)].length,3);
  for (const image of [...current.raw_images,...current.aux_images]) {
    assert(fs.existsSync(path.join(path.dirname(page),image)));
    images++;
  }
  for (const interior of current.interiors) {
    assert(nodes.interiors.innerHTML.includes(`活动 [${interior.start}, ${interior.stop})`));
    assert(nodes.interiors.innerHTML.includes(JSON.stringify(interior,null,2)));
    interiors++;
  }
  if (!current.interiors.length) {
    assert(nodes.interiors.innerHTML.includes('不据导数添加活动或阶段'));
    empty++;
  }
  states++;
}
assert.equal(states,12);
assert.equal(images,84);
assert.equal(interiors,21);
assert.equal(empty,3);
assert(nodes.denominator.textContent.includes('有观测2126日、缺测84日'));
console.log(JSON.stringify({status:'passed',states,images,interiors,empty,
  limitation:'DOM stub only; real browser rendering unverified'},null,2));
