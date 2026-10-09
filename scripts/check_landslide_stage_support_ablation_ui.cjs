/* Saved-page behavior in a DOM stub; not real browser rendering. */
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const page=path.resolve(process.argv[2]);
const html=fs.readFileSync(page,'utf8');
const payload=html.match(/<script[^>]*id="page-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload);
const nodes={};
const context=vm.createContext({document:{getElementById(id){
  if(!nodes[id]){nodes[id]={value:id==='method-select'?'rule':id==='condition-select'?'fixed_61':'0',
    textContent:id==='page-data'?payload:'',hidden:false,events:{},
    addEventListener(name,callback){this.events[name]=callback;}};}
  return nodes[id];
}}});
vm.runInContext(script,context);
let states=0,requestStates=0,skippedVisualStates=0;
for(let index=0;index<data.cases.length;index++){
  const current=data.cases[index];
  nodes['case-select'].value=String(index);
  for(const method of ['rule','xgboost','visual']){
    nodes['method-select'].value=method;
    for(const condition of data.summary.conditions){
      nodes['condition-select'].value=condition;
      nodes['condition-select'].events.change();
      assert(nodes['case-title'].textContent.includes(current.case_id));
      assert(nodes['case-title'].textContent.includes(method));
      assert.equal(nodes.comparison.src,current.plots[method]);
      assert(fs.existsSync(path.join(path.dirname(page),current.plots[method])));
      assert.deepEqual(JSON.parse(nodes['condition-detail'].textContent),{
        native_scores:current.native_scores[method],selected:current.summary[method][condition]});
      for(const interior of current.interiors){
        assert(nodes.interiors.innerHTML.includes(`活动 [${interior.start}, ${interior.stop})`));
        assert(nodes.interiors.innerHTML.includes(JSON.stringify(interior.methods[method][condition],null,2)));
      }
      assert.equal([...nodes['audit-inputs'].innerHTML.matchAll(/<img /g)].length,7);
      const request=method==='visual'?current.visual_request:null;
      assert.equal(nodes['request-detail'].hidden,!request);
      if(request){
        assert.equal(nodes.prompt.textContent,request.started.prompt);
        assert.equal(nodes.response.textContent,request.response.output);
        assert.equal([...nodes['native-inputs'].innerHTML.matchAll(/<img /g)].length,7);
        for(const image of request.images){assert(fs.existsSync(path.join(path.dirname(page),image)));}
        requestStates++;
      }else{
        assert.equal(nodes.prompt.textContent,'');
        assert.equal(nodes['native-inputs'].innerHTML,'');
        if(method==='visual'){assert(nodes['request-status'].textContent.includes('跳过'));skippedVisualStates++;}
      }
      states++;
    }
  }
}
assert.equal(states,180);
assert.equal(requestStates,45);
assert.equal(skippedVisualStates,15);
console.log(JSON.stringify({status:'passed',states,historical_request_states:requestStates,
  skipped_visual_states:skippedVisualStates,limitation:'DOM stub only; real browser rendering unverified'},null,2));
