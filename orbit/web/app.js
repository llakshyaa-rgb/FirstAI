const conversation=document.querySelector("#conversation");
const form=document.querySelector("#composer");
const input=document.querySelector("#message");
const send=document.querySelector("#send");
const presence=document.querySelector("#presence");
const device=document.querySelector("#device");
let busy=false;
function bubble(role,text,kind=""){
  const node=document.createElement("article");
  node.className="bubble "+role+(kind?" "+kind:"");
  const label=document.createElement("strong");
  label.textContent=role==="user"?"You":"Orbit";
  const p=document.createElement("p");p.textContent=text;
  node.append(label,p);conversation.append(node);conversation.scrollTop=conversation.scrollHeight;return p;
}
async function health(){
  try{const r=await fetch("/api/health");const d=await r.json();
    presence.textContent=d.status==="ready"?"Personal AI · Online":"Personal AI · Model not loaded";
    device.textContent=d.device||"";
    if(d.status!=="ready")bubble("orbit","The chat interface is running, but no trained checkpoint is loaded. "+(d.model_error||""));
  }catch{presence.textContent="Connection unavailable";}
}
async function follow(taskId){
  let after=0,finished=false;
  while(!finished){
    const r=await fetch("/api/events?task_id="+encodeURIComponent(taskId)+"&after="+after);
    if(!r.ok)throw new Error("Lost connection while checking task progress.");
    const d=await r.json();
    for(const event of d.events){
      after=Math.max(after,event.seq);
      if(event.kind==="message")bubble("orbit",event.message);
      else if(event.kind==="method_failed"||event.kind==="status"||event.kind==="generation_progress")bubble("orbit",event.message,"status");
    }
    const status=d.task.status;
    if(["completed","failed","needs_input","interrupted"].includes(status))finished=true;
    else await new Promise(resolve=>setTimeout(resolve,650));
  }
}
form.addEventListener("submit",async e=>{
  e.preventDefault();if(busy)return;
  const message=input.value.trim();if(!message)return;
  bubble("user",message);input.value="";busy=true;send.disabled=true;presence.textContent="Personal AI · Working";
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Request failed");
    await follow(d.task_id);
  }catch(err){bubble("orbit","I couldn't complete that request: "+err.message);}
  finally{busy=false;send.disabled=false;presence.textContent="Personal AI · Online";input.focus();}
});
health();
