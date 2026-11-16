#!/usr/bin/env python3
"""Stream, checkpoint and report progress for capability datasets."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
SOURCES={
"gsm8k":{"dataset":"openai/gsm8k","config":"main","split":"train","license":"MIT","url":"https://huggingface.co/datasets/openai/gsm8k"},
"arc":{"dataset":"allenai/ai2_arc","config":"ARC-Challenge","split":"train","license":"CC-BY-SA-4.0","url":"https://huggingface.co/datasets/allenai/ai2_arc"},
"squad":{"dataset":"rajpurkar/squad","config":"plain_text","split":"train","license":"CC-BY-SA-4.0","url":"https://huggingface.co/datasets/rajpurkar/squad"},
"mbpp":{"dataset":"google-research-datasets/mbpp","config":"sanitized","split":"train","license":"CC-BY-4.0","url":"https://huggingface.co/datasets/google-research-datasets/mbpp"}}
def convert(name,row,index,s):
    if name=="gsm8k":
        prompt="Solve this mathematics word problem. Show a clear, checkable solution.\n\n"+str(row.get("question","")).strip(); response=str(row.get("answer","")).strip()
    elif name=="arc":
        q=str(row.get("question","")).strip(); ch=row.get("choices") or {}; labels=ch.get("label") or []; texts=ch.get("text") or []; key=str(row.get("answerKey","")).strip(); answer=next((t for l,t in zip(labels,texts) if str(l)==key),"")
        if not q or not answer:return None
        prompt="Answer this science question using the choices.\n\n"+q+"\n"+"\n".join(f"{l}. {t}" for l,t in zip(labels,texts)); response=f"{key}. {answer}"
    elif name=="squad":
        ctx=str(row.get("context","")).strip(); q=str(row.get("question","")).strip(); answers=(row.get("answers") or {}).get("text") or []; response=str(answers[0]).strip() if answers else ""
        if not ctx or not q or not response:return None
        prompt="Answer only from this passage. If unsupported, say so.\n\nPassage:\n"+ctx+"\n\nQuestion: "+q
    else:
        task=str(row.get("text") or row.get("prompt") or "").strip(); response=str(row.get("code","")).strip()
        if not task or not response:return None
        tests=row.get("test_list") or []; prompt="Write a Python function for this task. Return code only.\n\nTask: "+task
        if tests: prompt+="\n\nTests:\n"+"\n".join(map(str,tests[:8]))
    if not prompt.strip() or not response.strip():return None
    return {"prompt":prompt,"response":response,"metadata":{"dataset_id":name,"dataset":s["dataset"],"config":s["config"],"split":s["split"],"license_declared_by_dataset_card":s["license"],"source_url":s["url"],"source_record_index":index}}
def download_one(name,limit,outdir):
    from datasets import load_dataset
    s=SOURCES[name]; target=outdir/f"{name}_train.jsonl"; partial=target.with_suffix(".jsonl.partial"); outdir.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size:
        n=sum(1 for x in target.open(encoding="utf-8") if x.strip()); print(f"[{name}] reuse completed {n:,}",flush=True); return {"dataset_id":name,"dataset":s["dataset"],"license":s["license"],"split":s["split"],"examples":n,"path":str(target)}
    saved=0; skip=0
    if partial.exists():
        with partial.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    saved+=1
                    try: skip=max(skip,int(json.loads(line)["metadata"]["source_record_index"])+1)
                    except Exception: pass
    print(f"[{name}] streaming {s['dataset']} | saved {saved:,}, resume source row {skip:,}",flush=True)
    ds=load_dataset(s["dataset"],s["config"],split=s["split"],streaming=True)
    count=saved; started=time.monotonic()
    with partial.open("a" if saved else "w",encoding="utf-8") as out:
        for idx,row in enumerate(ds):
            if idx<skip:continue
            if count>=limit:break
            item=convert(name,row,idx,s)
            if item is None:continue
            out.write(json.dumps(item,ensure_ascii=False)+"\n"); count+=1
            if count%250==0:
                out.flush(); print(f"[{name}] {count:,}/{limit:,} examples | source row {idx:,} | {count/max(time.monotonic()-started,.001):.1f}/s",flush=True)
        out.flush()
    if count<4:raise RuntimeError(f"{name}: only {count} usable examples; partial kept")
    partial.replace(target); print(f"[{name}] COMPLETE {count:,}",flush=True)
    return {"dataset_id":name,"dataset":s["dataset"],"license":s["license"],"split":s["split"],"examples":count,"path":str(target)}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",choices=list(SOURCES),default=list(SOURCES)); p.add_argument("--max-per-dataset",type=int,default=10000); p.add_argument("--output-dir",type=Path,default=Path("data/capabilities")); p.add_argument("--combined",type=Path,default=Path("data/instructions_capabilities.jsonl")); a=p.parse_args()
    if a.max_per_dataset<4:p.error("max-per-dataset must be >=4")
    results=[]
    for name in a.datasets: results.append(download_one(name,a.max_per_dataset,a.output_dir.resolve()))
    a.combined.parent.mkdir(parents=True,exist_ok=True)
    with a.combined.open("w",encoding="utf-8") as out:
        for r in results:
            with Path(r["path"]).open(encoding="utf-8") as src:
                for line in src:out.write(line)
    manifest={"schema_version":1,"datasets":results,"examples_total":sum(x["examples"] for x in results),"warning":"Review current dataset cards and license obligations before redistribution."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); print(json.dumps(manifest,indent=2),flush=True)
if __name__=="__main__":main()
