const conversation=document.querySelector("#conversation");
const form=document.querySelector("#composer");
const input=document.querySelector("#message");
const send=document.querySelector("#send");
const presence=document.querySelector("#presence");
const device=document.querySelector("#device");
let busy=false;
let livePartial=null;
let thinkingNode=null;
let modelReady=false;
let pendingAttachments=[];
const attachmentQueue=document.querySelector("#attachment-queue");
function renderAttachmentQueue(){
  if(!attachmentQueue)return;
  attachmentQueue.textContent="";
  for(const item of pendingAttachments){
    const chip=document.createElement("span");chip.className="attachment-chip";chip.textContent=item.name+" ";
    const remove=document.createElement("button");remove.type="button";remove.textContent="×";remove.setAttribute("aria-label","Remove "+item.name);
    remove.addEventListener("click",()=>{pendingAttachments=pendingAttachments.filter(x=>x.id!==item.id);renderAttachmentQueue();});
    chip.append(remove);attachmentQueue.append(chip);
  }
}

async function addUploadDelete(card,item){
  const remove=document.createElement("button");remove.type="button";remove.className="upload-delete";
  remove.textContent="Delete uploaded file";
  remove.addEventListener("click",async()=>{
    if(!confirm("Permanently delete "+item.name+" from Orbit's local upload storage?"))return;
    remove.disabled=true;
    try{
      const r=await fetch("/api/uploads?id="+encodeURIComponent(item.id),{method:"DELETE"});
      const d=await r.json();if(!r.ok)throw new Error(d.error||"Delete failed");
      card.remove();
      pendingAttachments=pendingAttachments.filter(x=>x.id!==item.id);renderAttachmentQueue();
      bubble("orbit","Deleted "+item.name+" from local upload storage.","status");
    }catch(err){bubble("orbit","I couldn't delete that upload: "+err.message,"status");remove.disabled=false;}
  });
  card.append(remove);
}

function safeHttpUrl(value){
  try{const url=new URL(String(value),window.location.href);return url.protocol==="http:"||url.protocol==="https:"?url.href:"";}
  catch{return "";}
}
function bubble(role,text,kind=""){
  const node=document.createElement("article");
  node.className="bubble "+role+(kind?" "+kind:"");
  const label=document.createElement("strong");
  label.textContent=role==="user"?"You":"Orbit";
  const p=document.createElement("p");p.textContent=text;
  node.append(label,p);conversation.append(node);conversation.scrollTop=conversation.scrollHeight;return p;
}
function showThinking(){
  if(thinkingNode?.isConnected)return;
  const node=document.createElement("article");node.className="bubble orbit thinking";
  const label=document.createElement("strong");label.textContent="Orbit";
  const indicator=document.createElement("span");indicator.className="thinking-indicator";
  indicator.setAttribute("role","status");indicator.setAttribute("aria-label","Thinking");
  const mark=document.createElement("span");mark.className="thinking-mark";mark.setAttribute("aria-hidden","true");
  const dots=document.createElement("span");dots.className="thinking-dots";dots.setAttribute("aria-hidden","true");
  dots.textContent="•••";indicator.append(mark,dots);node.append(label,indicator);
  conversation.append(node);conversation.scrollTop=conversation.scrollHeight;thinkingNode=node;
}
function hideThinking(){
  thinkingNode?.remove();thinkingNode=null;
}
async function health(){
  try{const r=await fetch("/api/health");const d=await r.json();
    modelReady=d.status==="ready";
    presence.textContent=modelReady?"Personal AI · Online":"Local tools online · language model not loaded";
    device.textContent=d.device||"";
    const imageStatus=document.querySelector("#image-status");
    if(imageStatus)imageStatus.textContent=[
      d.image_backend==="diffusers"?(d.image_backend_connected?"Diffusers ready · "+(d.image_model||"image model"):"Diffusers unavailable"):(d.image_backend_connected?"ComfyUI online":(d.image_workflow_configured?"Image workflow · backend offline":"Image generation not configured")),
      d.image_search_configured?"Web image search ready":"Image search key missing",
      d.vision_configured?"Vision model configured":"Vision model not configured"
    ].join(" · ");
    if(d.status!=="ready")bubble("orbit","The chat interface is running, but no trained checkpoint is loaded. "+(d.model_error||""));
  }catch{presence.textContent="Connection unavailable";}
}
const imageStudio=document.querySelector("#image-studio");
const imageStudioState=document.querySelector("#image-studio-state");
const imageStudioContent=document.querySelector("#image-studio-content");
function setImageStudioLoading(){imageStudio.hidden=false;imageStudioState.textContent="Generating…";imageStudioContent.innerHTML='<span class="image-studio-spinner" role="status"></span><p>Orbit is creating your image. Chat is paused.</p>';}
function setImageStudioError(message){imageStudio.hidden=false;imageStudioState.textContent="Generation failed";imageStudioContent.textContent="";const p=document.createElement("p");p.className="image-studio-error";p.textContent=message||"Image generation failed.";imageStudioContent.append(p);}
function showGeneratedImage(url){const safe=safeHttpUrl(url);if(!safe){setImageStudioError("Invalid image URL returned.");return;}imageStudio.hidden=false;imageStudioState.textContent="Complete";imageStudioContent.textContent="";const img=document.createElement("img");img.src=safe;img.alt="Image generated by Orbit";img.className="image-studio-image";img.addEventListener("error",()=>setImageStudioError("Preview could not load the generated image."));const a=document.createElement("a");a.href=safe;a.download="orbit-image.png";a.textContent="Save to Downloads";a.className="image-studio-save";imageStudioContent.append(img,a);}
async function follow(taskId,afterStart=0,isImageTask=false){
  let after=afterStart,finished=false,finalStatus="";
  while(!finished){
    const r=await fetch("/api/events?task_id="+encodeURIComponent(taskId)+"&after="+after);
    if(!r.ok)throw new Error("Lost connection while checking task progress.");
    const d=await r.json();
    for(const event of d.events){
      after=Math.max(after,event.seq);
      if(event.kind==="message"){
        if(isImageTask)continue;
        hideThinking();
        const suggested=event.details?.coding_result?.suggested_preview_path;
        if(suggested&&previewProjectPath)previewProjectPath.value=suggested==="."?"":suggested;
        if(livePartial){livePartial.textContent=event.message;livePartial=null;}else bubble("orbit",event.message);
      }
      else if(event.kind==="partial_reset"){
        if(isImageTask)continue;
        if(livePartial){livePartial.parentElement?.remove();livePartial=null;}
        showThinking();
      }
      else if(event.kind==="partial"){
        if(isImageTask)continue;
        hideThinking();
        if(!livePartial)livePartial=bubble("orbit",event.message,"partial");
        else livePartial.textContent=event.message;
        conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="approval_required"){
        const details=event.details||{};
        const node=document.createElement("article");node.className="bubble orbit approval-card";
        const label=document.createElement("strong");label.textContent="Orbit · Approval required";
        const summary=document.createElement("p");summary.textContent=details.summary||event.message||"Orbit proposes a software change.";
        const files=document.createElement("small");files.textContent="Files: "+((details.files||[]).join(", ")||"Not specified");
        const preview=document.createElement("pre");preview.className="approval-preview";
        preview.textContent=details.proposal_preview||"No patch preview was supplied. Review the file list and phase summary carefully.";
        const actions=document.createElement("div");actions.className="approval-actions";
        const approve=document.createElement("button");approve.type="button";approve.textContent="Approve this phase";
        const deny=document.createElement("button");deny.type="button";deny.textContent="Deny";
        const status=document.createElement("small");status.textContent="No files will be changed until you approve.";
        async function decide(approved){
          approve.disabled=true;deny.disabled=true;status.textContent=approved?"Sending approval…":"Sending denial…";
          try{
            const response=await fetch("/api/coding/approval",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({approval_id:details.approval_id,approved})});
            const result=await response.json();if(!response.ok)throw new Error(result.error||"Decision failed");
            status.textContent=approved?"Approved. Orbit may now apply this phase.":"Denied. Orbit will not apply this phase.";
          }catch(error){status.textContent="Decision wasn't confirmed: "+error.message;approve.disabled=false;deny.disabled=false;}
        }
        approve.addEventListener("click",()=>decide(true));
        deny.addEventListener("click",()=>decide(false));
        actions.append(approve,deny);node.append(label,summary,files,preview,actions,status);
        conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="code_change"){
        const change=event.details.change||{};
        const node=document.createElement("article");node.className="bubble orbit code-change";
        const label=document.createElement("strong");label.textContent="Orbit · File updated";
        const p=document.createElement("p");p.textContent=change.path||"Project file";
        const backup=document.createElement("small");backup.textContent=change.backup?"Backup saved: "+change.backup:"New file created";
        node.append(label,p,backup);conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="image"){showGeneratedImage(event.details?.url);}
      else if(event.kind==="video"||event.kind==="audio"){
        const node=document.createElement("article");node.className="bubble orbit generated-media";
        const label=document.createElement("strong");label.textContent=event.kind==="video"?"Orbit · Video":"Orbit · Audio";
        const url=safeHttpUrl(event.details.url);
        if(url){
          const player=document.createElement(event.kind==="video"?"video":"audio");
          player.src=url;player.controls=true;player.preload="metadata";
          if(event.kind==="video"){player.className="generated-video";player.playsInline=true;}
          else player.className="generated-audio";
          const link=document.createElement("a");link.href=url;link.download=event.details.filename||("orbit-generated."+ (event.kind==="video"?"mp4":"wav"));link.textContent="Download "+event.kind;link.className="file-link";
          node.append(label,player,link);
        }else{
          const p=document.createElement("p");p.textContent="The specialist returned an invalid media URL.";node.append(label,p);
        }
        conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="image_results"){
        const node=document.createElement("article");node.className="bubble orbit image-search-message";
        const label=document.createElement("strong");label.textContent="Orbit · Image research";
        node.append(label);
        for(const item of (event.details.results||[])){
          const card=document.createElement("div");card.className="search-card";
          const link=document.createElement("a");link.href=safeHttpUrl(item.page_url);link.target="_blank";link.rel="noopener noreferrer";
          const img=document.createElement("img");img.src=safeHttpUrl(item.thumbnail_url);img.alt=item.title||"Image result";img.loading="lazy";link.append(img);
          if(!link.href||!img.src)continue;
          const title=document.createElement("p");title.textContent=item.title||"Image result";
          const source=document.createElement("small");source.textContent=item.source||"Open source page";
          card.append(link,title,source);node.append(card);
        }
        if(!(event.details.results||[]).length){const p=document.createElement("p");p.textContent="No image results were returned.";node.append(p);}
        conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="artifact"){
        const artifact=event.details.artifact||{};
        const node=document.createElement("article");node.className="bubble orbit artifact-result";
        const label=document.createElement("strong");label.textContent="Orbit · Finished design";
        const p=document.createElement("p");p.textContent=artifact.title+" · "+String(artifact.format||"file").toUpperCase()+" · "+Math.ceil((artifact.size||0)/1024)+" KB";
        const link=document.createElement("a");link.href=artifact.url;link.download=artifact.name||"orbit-design";link.textContent="Open / download "+(artifact.name||"file");link.className="file-link";
        node.append(label,p,link);conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
      }
      else if(event.kind==="memory_saved")bubble("orbit","Memory updated: "+((event.details.memories||[]).map(x=>x.key+": "+x.value).join("; ")),"status");
      else if(event.kind==="method_failed"){if(isImageTask)setImageStudioError(event.message);else bubble("orbit",event.message,"status");}
    }
    const status=d.task.status;finalStatus=status;
    if(["completed","failed","needs_input","interrupted"].includes(status))finished=true;
    else await new Promise(resolve=>setTimeout(resolve,650));
  }
  hideThinking();
  if(isImageTask&&["failed","needs_input","interrupted"].includes(finalStatus)){if(imageStudioState.textContent!=="Generation failed")setImageStudioError("Image task stopped: "+finalStatus);return;}
  if(["failed","needs_input","interrupted"].includes(finalStatus)){
    const node=document.createElement("article");node.className="bubble orbit task-retry";
    const label=document.createElement("strong");label.textContent="Task stopped";
    const p=document.createElement("p");p.textContent="The task state is preserved. Orbit can retry it using the same request.";
    const retry=document.createElement("button");retry.type="button";retry.textContent="Retry task";
    retry.addEventListener("click",async()=>{
      retry.disabled=true;retry.textContent="Retrying…";
      try{
        const r=await fetch("/api/tasks/retry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task_id:taskId})});
        const d=await r.json();if(!r.ok)throw new Error(d.error||"Retry request failed");
        await follow(taskId,after);
      }catch(err){p.textContent="Retry could not start: "+err.message;retry.disabled=false;retry.textContent="Retry task";}
    });
    node.append(label,p,retry);conversation.append(node);conversation.scrollTop=conversation.scrollHeight;
  }
}
form.addEventListener("submit",async e=>{
e.preventDefault();if(busy)return;
const message=input.value.trim();if(!message)return;
const imageTask=/\b(generate|create|draw|make|render|paint|illustrate)\b[\s\S]{0,100}\b(image|picture|wallpaper|artwork|illustration|portrait|logo|poster)\b/i.test(message)||/\b(image|picture|wallpaper|artwork|illustration|portrait|logo|poster)\b[\s\S]{0,80}\b(generate|create|draw|make|render|paint)\b/i.test(message);
livePartial=null;if(imageTask){setImageStudioLoading();hideThinking();}else{bubble("user",message);showThinking();}
input.value="";busy=true;send.disabled=true;input.disabled=true;if(imageTask){send.textContent="Generating…";presence.textContent="Personal AI · Creating image";}else presence.textContent="Personal AI · Working";
try{const endpoint=imageTask?"/api/image":"/api/chat";const payload=imageTask?{prompt:message,conversation_id:activeConversationId}:{message,attachments:pendingAttachments.map(x=>x.id),conversation_id:activeConversationId};const r=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.error||"Request failed");pendingAttachments=[];renderAttachmentQueue();await follow(d.task_id,0,imageTask);await loadHistory();}
catch(err){hideThinking();if(imageTask)setImageStudioError(err.message);else bubble("orbit","I couldn't complete that request: "+err.message);}
finally{hideThinking();busy=false;send.disabled=false;input.disabled=false;send.textContent="Send";presence.textContent=modelReady?"Personal AI · Online":"Local tools online · language model not loaded";input.focus();}
});
let activeConversationId=null;
const historyList=document.querySelector("#history-list");
const historyCount=document.querySelector("#history-count");
function clearConversation(){
  conversation.textContent="";
  livePartial=null;
}
function renderSavedConversation(item){
  clearConversation();
  activeConversationId=item?.id||null;
  if(!item){
    const node=document.createElement("article");node.className="bubble orbit";
    const label=document.createElement("strong");label.textContent="Orbit";
    const p=document.createElement("p");p.textContent="New chat. What would you like to work on?";
    node.append(label,p);conversation.append(node);
  }else{
    for(const entry of item.messages||[]){
      if(entry.kind==="image"||entry.kind==="video"||entry.kind==="audio"){
        const node=document.createElement("article");node.className="bubble orbit generated-media";
        const label=document.createElement("strong");label.textContent="Orbit · "+entry.kind.charAt(0).toUpperCase()+entry.kind.slice(1);
        const url=safeHttpUrl(entry.url);
        if(url){
          const media=document.createElement(entry.kind==="image"?"img":entry.kind);
          media.src=url;
          if(entry.kind==="image"){media.alt="Previously generated image";media.loading="lazy";media.className="generated-image";}
          else{media.controls=true;media.preload="metadata";media.className=entry.kind==="video"?"generated-video":"generated-audio";if(entry.kind==="video")media.playsInline=true;}
          node.append(label,media);
          const link=document.createElement("a");link.href=url;link.download=entry.filename||("orbit-generated."+ (entry.kind==="video"?"mp4":entry.kind==="audio"?"wav":"png"));link.textContent="Download "+entry.kind;link.className="file-link";node.append(link);
        }
        conversation.append(node);
      }else bubble(entry.role==="user"?"user":"orbit",entry.content||"");
    }
    if(!(item.messages||[]).length){
      const node=document.createElement("article");node.className="bubble orbit";
      const label=document.createElement("strong");label.textContent="Orbit";
      const p=document.createElement("p");p.textContent="Continue this conversation.";
      node.append(label,p);conversation.append(node);
    }
  }
  conversation.scrollTop=conversation.scrollHeight;
  for(const button of historyList?.querySelectorAll("[data-conversation-id]")||[]){
    button.classList.toggle("active",button.dataset.conversationId===activeConversationId);
  }
}
async function loadHistory(selectId=null){
  try{
    const r=await fetch("/api/history");if(!r.ok)return;
    const d=await r.json();
    if(historyList){
      historyList.textContent="";
      const items=d.conversations||[];
      if(historyCount)historyCount.textContent=items.length+" saved";
      for(const item of items){
        const row=document.createElement("div");row.className="history-row";
        const button=document.createElement("button");button.type="button";button.className="history-item";
        button.dataset.conversationId=item.id;
        const title=document.createElement("span");title.className="history-title";title.textContent=item.title||"Untitled chat";
        const meta=document.createElement("small");meta.textContent=(item.status==="completed"?"Completed":item.status==="failed"?"Failed":item.status==="running"?"In progress":"Saved")+" · "+new Date(item.created_at).toLocaleDateString();
        button.append(title,meta);
        button.addEventListener("click",async()=>{
          try{
            const response=await fetch("/api/history?id="+encodeURIComponent(item.id));
            const saved=await response.json();if(!response.ok||!saved.conversation)throw new Error("Saved chat unavailable");
            renderSavedConversation(saved.conversation);
          }catch(error){bubble("orbit","Couldn't open that saved chat: "+error.message,"status");}
        });
        const remove=document.createElement("button");remove.type="button";remove.className="history-delete";remove.textContent="Delete";
        remove.setAttribute("aria-label","Delete conversation "+(item.title||"Untitled chat"));
        remove.addEventListener("click",async()=>{
          if(!confirm("Permanently delete this conversation and all its messages? This cannot be undone."))return;
          remove.disabled=true;
          try{
            const response=await fetch("/api/conversations?id="+encodeURIComponent(item.id),{method:"DELETE"});
            const result=await response.json();if(!response.ok)throw new Error(result.error||"Delete failed");
            if(activeConversationId===item.id){
              activeConversationId=null;
              await createNewConversation(false);
            }else await loadHistory();
          }catch(error){remove.disabled=false;bubble("orbit","Couldn't delete that conversation: "+error.message,"status");}
        });
        row.append(button,remove);historyList.append(row);
      }
      if(!items.length){const empty=document.createElement("p");empty.className="history-empty";empty.textContent="Your conversations will appear here.";historyList.append(empty);}
    }
    const targetId=selectId||activeConversationId||((d.conversations||[])[0]?.id);
    if(targetId){
      const response=await fetch("/api/history?id="+encodeURIComponent(targetId));
      const saved=await response.json();
      if(response.ok&&saved.conversation)renderSavedConversation(saved.conversation);
    }else if(!activeConversationId){
      await createNewConversation(false);
    }
    conversation.scrollTop=conversation.scrollHeight;
  }catch{}
}
async function createNewConversation(showWelcome=true){
  const r=await fetch("/api/conversations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:"New chat"})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Could not create a new conversation");
  activeConversationId=d.conversation_id;
  renderSavedConversation({id:activeConversationId,messages:[]});
  await loadHistory();
  if(showWelcome)input.focus();
}
document.querySelector("#new-chat")?.addEventListener("click",async()=>{
  try{await createNewConversation(true);}
  catch(error){bubble("orbit","Couldn't start a new chat: "+error.message,"status");}
});

health();
loadHistory();

const memoryKey=document.querySelector("#memory-key");
const memoryValue=document.querySelector("#memory-value");
const memorySave=document.querySelector("#memory-save");
const memoryList=document.querySelector("#memory-list");
function renderMemories(items){
  memoryList.textContent="";
  for(const item of items||[]){
    const row=document.createElement("div");row.className="memory-row";
    const p=document.createElement("p");p.className="memory-item";p.textContent=item.key+": "+item.value;
    const remove=document.createElement("button");remove.type="button";remove.className="memory-delete";remove.textContent="Delete";
    remove.addEventListener("click",async()=>{
      const r=await fetch("/api/memory?key="+encodeURIComponent(item.key),{method:"DELETE"});
      const d=await r.json();if(r.ok)renderMemories(d.memories);
    });
    row.append(p,remove);memoryList.append(row);
  }
}
async function loadMemories(){
  try{const r=await fetch("/api/memory");const d=await r.json();renderMemories(d.memories);}catch{}
}
memorySave?.addEventListener("click",async()=>{
  const key=memoryKey.value.trim(),value=memoryValue.value.trim();if(!key||!value)return;
  memorySave.disabled=true;
  try{const r=await fetch("/api/memory",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key,value})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Could not save preference");
    memoryKey.value="";memoryValue.value="";renderMemories(d.memories);
  }catch(err){bubble("orbit","I couldn't save that preference: "+err.message,"status");}
  finally{memorySave.disabled=false;}
});
loadMemories();

const imageForm=document.querySelector("#image-composer");
const imagePrompt=document.querySelector("#image-prompt");
const imageNegative=document.querySelector("#image-negative");
const imageSend=document.querySelector("#image-send");
imageForm?.addEventListener("submit",async e=>{
  e.preventDefault();const prompt=imagePrompt.value.trim();if(!prompt||busy)return;
  livePartial=null;
  busy=true;imageSend.disabled=true;presence.textContent="Personal AI · Creating image";
  bubble("user","Create image: "+prompt);
  try{
    const r=await fetch("/api/image",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt,negative_prompt:imageNegative.value})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Image request failed");
    await follow(d.task_id);
  }catch(err){bubble("orbit","I couldn't generate that image: "+err.message);}
  finally{busy=false;imageSend.disabled=false;presence.textContent=modelReady?"Personal AI · Online":"Local tools online · language model not loaded";}
});

const uploadForm=document.querySelector("#upload-form");
const fileInput=document.querySelector("#file-input");
const uploadSend=document.querySelector("#upload-send");
const uploadResults=document.querySelector("#upload-results");
let uploadInProgress=false;
async function uploadSelectedFiles(){
  if(uploadInProgress)return;
  const files=[...(fileInput?.files||[])];
  if(!files.length)return;
  uploadInProgress=true;uploadSend.disabled=true;
  const originalLabel=uploadSend.textContent;uploadSend.textContent="Uploading…";
  const body=new FormData();for(const file of files)body.append("files",file,file.name);
  try{
    const r=await fetch("/api/upload",{method:"POST",body});
    const d=await r.json().catch(()=>({error:"The server returned an unreadable response (HTTP "+r.status+")."}));
    if(!r.ok)throw new Error(d.error||"Upload failed");
    if(!Array.isArray(d.files)||!d.files.length)throw new Error("The server returned no uploaded files.");
    let queued=0;
    for(const item of d.files){
      const card=document.createElement("article");card.className="upload-card";
      const heading=document.createElement("strong");heading.textContent=item.name+" · "+Math.ceil(item.size/1024)+" KB";card.append(heading);
      const fileLink=document.createElement("a");fileLink.href=item.url;fileLink.download=item.name;fileLink.textContent="Open / download";fileLink.className="file-link";card.append(fileLink);
      if(item.category==="media" && item.content_type?.startsWith("audio/")){
        const player=document.createElement("audio");player.controls=true;player.src=item.url;card.append(player);
      }else if(item.category==="media" && item.content_type?.startsWith("video/")){
        const player=document.createElement("video");player.controls=true;player.src=item.url;player.className="uploaded-video";card.append(player);
      }
      if(item.category==="image"){
        const img=document.createElement("img");img.src=item.url;img.alt=item.name;img.loading="lazy";img.className="uploaded-image";card.append(img);
      }
      if(item.category==="image" || (item.category==="media" && /^(audio|video)\//.test(item.content_type||""))){
        const analyze=document.createElement("button");analyze.type="button";
        analyze.textContent=item.content_type?.startsWith("audio/")?"Transcribe / understand audio":item.content_type?.startsWith("video/")?"Analyse video":"Understand image";
        analyze.addEventListener("click",async()=>{
          analyze.disabled=true;const original=analyze.textContent;analyze.textContent="Processing locally…";
          try{
            const response=await fetch("/api/vision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file_id:item.id,prompt:"Analyse this uploaded media carefully. For audio, transcribe speech and identify language. For video, summarise sampled scenes and available audio. For images, describe visible content and readable text. State uncertainty clearly."})});
            const result=await response.json();if(!response.ok)throw new Error(result.error||"Media analysis failed");
            bubble("orbit",result.kind.toUpperCase()+" · "+result.name+"\n\n"+result.description);
          }catch(err){bubble("orbit","Media analysis isn't available: "+err.message,"status");}
          finally{analyze.disabled=false;analyze.textContent=original;}
        });card.append(analyze);
      }
      if(item.category!=="image"){
        const meta=document.createElement("p");meta.textContent=item.category+" · "+item.content_type;card.append(meta);
      }
      if(item.text){
        const use=document.createElement("button");use.type="button";use.textContent="Use extracted text in chat";
        use.addEventListener("click",()=>{input.value="Please work with this uploaded file ("+item.name+"):\n\n"+item.text;input.focus();});
        card.append(use);
        const details=document.createElement("details");const summary=document.createElement("summary");summary.textContent="Extracted text preview";
        const pre=document.createElement("pre");pre.textContent=item.text.slice(0,3000);details.append(summary,pre);card.append(details);
      }
      if(pendingAttachments.length<5 && !pendingAttachments.some(x=>x.id===item.id)){
        pendingAttachments.push({id:item.id,name:item.name});queued++;renderAttachmentQueue();
      }
      const attach=document.createElement("button");attach.type="button";
      const updateAttachLabel=()=>{attach.textContent=pendingAttachments.some(x=>x.id===item.id)?"Queued for next chat · Remove":"Attach to next chat";};
      attach.addEventListener("click",()=>{
        if(pendingAttachments.some(x=>x.id===item.id)){
          pendingAttachments=pendingAttachments.filter(x=>x.id!==item.id);
        }else{
          if(pendingAttachments.length>=5){bubble("orbit","A maximum of five files can be attached to one message.","status");return;}
          pendingAttachments.push({id:item.id,name:item.name});
        }
        renderAttachmentQueue();updateAttachLabel();
      });
      updateAttachLabel();card.append(attach);
      await addUploadDelete(card,item);uploadResults.prepend(card);
    }
    if(queued)bubble("orbit",d.files.length+" file"+(d.files.length===1?"":"s")+" uploaded and queued for your next message.","status");
    if(queued<d.files.length)bubble("orbit","Some files were uploaded but not queued because a message can contain at most five attachments.","status");
    fileInput.value="";
  }catch(err){bubble("orbit","I couldn't upload those files: "+err.message,"status");}
  finally{uploadInProgress=false;uploadSend.disabled=false;uploadSend.textContent=originalLabel;}
}
uploadForm?.addEventListener("submit",e=>{e.preventDefault();uploadSelectedFiles();});
// Selecting a file uploads and queues it immediately; the button remains as a fallback.
fileInput?.addEventListener("change",uploadSelectedFiles);
const imageSearchForm=document.querySelector("#image-search-form");
const imageSearchQuery=document.querySelector("#image-search-query");
const imageSearchResults=document.querySelector("#image-search-results");
const showMature=document.querySelector("#show-mature");
imageSearchForm?.addEventListener("submit",async e=>{
  e.preventDefault();const query=imageSearchQuery.value.trim();if(!query)return;
  imageSearchResults.textContent="Searching…";
  try{
    const r=await fetch("/api/image-search?q="+encodeURIComponent(query)+"&safe="+(showMature?.checked?"off":"strict"));
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Search failed");
    imageSearchResults.textContent="";
    if(d.may_include_mature){const warning=document.createElement("p");warning.className="media-note";warning.textContent="SafeSearch is off; results may include mature material.";imageSearchResults.append(warning);}
    const results=(d.results||[]);
    for(const item of results){
      const card=document.createElement("article");card.className="search-card";
      const link=document.createElement("a");link.href=item.page_url;link.target="_blank";link.rel="noopener noreferrer";
      const img=document.createElement("img");img.src=item.thumbnail_url;img.alt=item.title;img.loading="lazy";link.append(img);
      const title=document.createElement("p");title.textContent=item.title;
      const source=document.createElement("small");source.textContent=item.source||"Open source page";
      card.append(link,title,source);

      imageSearchResults.append(card);
    }
    if(!results.length)imageSearchResults.textContent="No results returned for this search.";
  }catch(err){imageSearchResults.textContent=err.message;}
});

async function restoreUploads(){
  try{
    const r=await fetch("/api/uploads");if(!r.ok)return;
    const d=await r.json();
    for(const item of d.files||[]){
      const card=document.createElement("article");card.className="upload-card";
      const heading=document.createElement("strong");heading.textContent=item.name+" · "+Math.ceil(item.size/1024)+" KB";card.append(heading);
      const fileLink=document.createElement("a");fileLink.href=item.url;fileLink.download=item.name;fileLink.textContent="Open / download";fileLink.className="file-link";card.append(fileLink);
      if(item.category==="media" && item.content_type?.startsWith("audio/")){
        const player=document.createElement("audio");player.controls=true;player.src=item.url;card.append(player);
      }else if(item.category==="media" && item.content_type?.startsWith("video/")){
        const player=document.createElement("video");player.controls=true;player.src=item.url;player.className="uploaded-video";card.append(player);
      }
      if(item.category==="image"){
        const img=document.createElement("img");img.src=item.url;img.alt=item.name;img.loading="lazy";img.className="uploaded-image";card.append(img);
      }
      if(item.category==="image" || (item.category==="media" && /^(audio|video)\//.test(item.content_type||""))){
        const analyze=document.createElement("button");analyze.type="button";
        analyze.textContent=item.content_type?.startsWith("audio/")?"Transcribe / understand audio":item.content_type?.startsWith("video/")?"Analyse video":"Understand image";
        analyze.addEventListener("click",async()=>{
          analyze.disabled=true;const original=analyze.textContent;analyze.textContent="Processing locally…";
          try{const response=await fetch("/api/vision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file_id:item.id,prompt:"Analyse this uploaded media carefully. For audio, transcribe speech and identify language. For video, summarise sampled scenes and available audio. For images, describe visible content and readable text. State uncertainty clearly."})});const result=await response.json();if(!response.ok)throw new Error(result.error||"Media analysis failed");bubble("orbit",result.kind.toUpperCase()+" · "+result.name+"\\n\\n"+result.description);}
          catch(err){bubble("orbit","Media analysis isn't available: "+err.message,"status");}
          finally{analyze.disabled=false;analyze.textContent=original;}
        });card.append(analyze);
      }
      const attach=document.createElement("button");attach.type="button";attach.textContent="Attach to next chat";
      attach.addEventListener("click",()=>{
        if(pendingAttachments.length>=5){bubble("orbit","A maximum of five files can be attached to one message.","status");return;}
        if(!pendingAttachments.some(x=>x.id===item.id)){pendingAttachments.push({id:item.id,name:item.name});renderAttachmentQueue();}
      });
      card.append(attach);await addUploadDelete(card,item);uploadResults.append(card);
    }
  }catch{}
}
restoreUploads();

const designForm=document.querySelector("#design-form");
const designTitle=document.querySelector("#design-title");
const designBrief=document.querySelector("#design-brief");
const designFormat=document.querySelector("#design-format");
const designTheme=document.querySelector("#design-theme");
const designSend=document.querySelector("#design-send");
designForm?.addEventListener("submit",async e=>{
  e.preventDefault();
  if(busy)return;
  const title=designTitle.value.trim(),brief=designBrief.value.trim();
  if(!title||!brief)return;
  busy=true;designSend.disabled=true;presence.textContent="Personal AI · Designing";
  bubble("user","Design: "+title+" ("+designFormat.value.toUpperCase()+")");
  try{
    const r=await fetch("/api/design",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({title,brief,format:designFormat.value,theme:designTheme.value})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Design request failed");
    await follow(d.task_id);
  }catch(err){bubble("orbit","I couldn't create that design: "+err.message,"status");}
  finally{busy=false;designSend.disabled=false;presence.textContent=modelReady?"Personal AI · Online":"Local tools online · language model not loaded";}
});


const previewStart=document.querySelector("#preview-start");
const previewStop=document.querySelector("#preview-stop");
const previewInspect=document.querySelector("#preview-inspect");
const previewInspectionOutput=document.querySelector("#preview-inspection-output");
const previewState=document.querySelector("#preview-state");
const previewFrame=document.querySelector("#app-preview");
const previewOpen=document.querySelector("#preview-open");
const previewProjectPath=document.querySelector("#preview-project-path");
const previewCloudOpen=document.querySelector("#preview-cloud-open");
const previewLogOutput=document.querySelector("#preview-log-output");
function showPreview(data){
  const running=!!data.running;
  const cloud=!!data.remote_preview_url;
  previewState.textContent=running?"Running on this device · "+data.port:(cloud?"Connected to configured cloud preview":"No project running");
  previewStart.disabled=running;previewStop.disabled=!running;if(previewInspect)previewInspect.disabled=!running;
  previewOpen.hidden=!running;
  if(previewCloudOpen){previewCloudOpen.hidden=!cloud;previewCloudOpen.href=data.remote_preview_url||"#";}
  const target=running?data.url+"/":(cloud?data.remote_preview_url:"");
  if(cloud&&!running)previewFrame.sandbox.remove("allow-forms");
  else previewFrame.sandbox.add("allow-forms");
  if(target){previewFrame.hidden=false;if(previewFrame.src!==new URL(target,location.href).href)previewFrame.src=target;}
  else{previewFrame.hidden=true;previewFrame.src="about:blank";}
  if(running)previewOpen.href=data.url;else previewOpen.removeAttribute("href");
  if(data.remote_preview_error)previewState.textContent=data.remote_preview_error;
}
async function refreshPreview(){
  try{
    const r=await fetch("/api/preview/status");
    if(r.ok)showPreview(await r.json());
    const logs=await fetch("/api/preview/logs");
    if(logs.ok){const d=await logs.json();if(previewLogOutput)previewLogOutput.textContent=d.text||"No logs yet.";}
  }catch{}
}
previewStart?.addEventListener("click",async()=>{
  previewStart.disabled=true;previewState.textContent="Starting development server…";
  try{const r=await fetch("/api/preview/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_path:previewProjectPath?.value.trim()||null})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Preview could not start");showPreview(d);
  }catch(err){previewState.textContent=err.message;previewStart.disabled=false;}
});
previewInspect?.addEventListener("click",async()=>{
  previewInspect.disabled=true;
  if(previewInspectionOutput)previewInspectionOutput.textContent="Inspecting the running app without clicking controls or submitting forms…";
  try{
    const r=await fetch("/api/preview/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Browser inspection failed");
    if(previewInspectionOutput)previewInspectionOutput.textContent=JSON.stringify(d,null,2);
  }catch(err){if(previewInspectionOutput)previewInspectionOutput.textContent=err.message;}
  finally{previewInspect.disabled=false;}
});
previewStop?.addEventListener("click",async()=>{
  previewStop.disabled=true;
  try{const r=await fetch("/api/preview/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json();if(!r.ok)throw new Error(d.error||"Could not stop preview");showPreview(d);
  }catch(err){previewState.textContent=err.message;previewStop.disabled=false;}
});
refreshPreview();
setInterval(refreshPreview, 4000);
