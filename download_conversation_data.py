#!/usr/bin/env python3
"""Stream and checkpoint assistant turns from UltraChat."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
DATASET="HuggingFaceH4/ultrachat_200k"; SPLIT="train_sft"
def messages(row):
    out=[]
    for x in row.get("messages") or row.get("conversation") or []:
        if not isinstance(x,dict):continue
        role=str(x.get("role") or x.get("from") or "").lower()
        body=x.get("content",x.get("value"))
        role={"human":"user","user":"user","assistant":"assistant","gpt":"assistant"}.get(role)
        if role and isinstance(body,str) and body.strip():out.append((role,body.strip()))
    return out
def examples(msgs,index,max_context):
    prior=[]
    for role,body in msgs:
        if role=="assistant" and any(r=="user" for r,_ in prior) and len(body)<=1800:
            prompt="\n".join(("Owner: " if r=="user" else "Orbit: ")+t for r,t in prior[-12:])
            prompt=prompt[-max_context:]
            response="Orbit: "+body
            digest=hashlib.sha256((prompt+"\0"+response).encode()).hexdigest()
            yield {"prompt":prompt,"response":response,"metadata":{"dataset_id":"ultrachat_200k","dataset":DATASET,"split":SPLIT,"source_url":"https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k","license_declared_by_dataset_card":"MIT","source_record_index":index,"turn_hash":digest}}
        prior.append((role,body))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=20000); p.add_argument("--output",type=Path,default=Path("data/instructions_conversations.jsonl")); p.add_argument("--max-context-chars",type=int,default=2500); a=p.parse_args()
    from datasets import load_dataset
    a.output.parent.mkdir(parents=True,exist_ok=True); partial=a.output.with_suffix(a.output.suffix+".partial")
    if a.output.exists() and a.output.stat().st_size:
        print(f"[UltraChat] already complete: {a.output}",flush=True); return
    saved=0; skip=0; seen=set()
    if partial.exists():
        with partial.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    saved+=1
                    try:skip=max(skip,int(json.loads(line)["metadata"]["source_record_index"])+1)
                    except Exception:pass
    print(f"[UltraChat] streaming {DATASET}/{SPLIT}; resume {saved:,} turns at source row {skip:,}",flush=True)
    ds=load_dataset(DATASET,split=SPLIT,streaming=True); count=saved; started=time.monotonic()
    with partial.open("a" if saved else "w",encoding="utf-8") as out:
        for idx,row in enumerate(ds):
            if idx<skip:continue
            if count>=a.limit:break
            for item in examples(messages(row),idx,a.max_context_chars):
                h=item["metadata"]["turn_hash"]
                if h in seen:continue
                seen.add(h); out.write(json.dumps(item,ensure_ascii=False)+"\n"); count+=1
                if count%500==0:
                    out.flush(); print(f"[UltraChat] {count:,}/{a.limit:,} turns | row {idx:,} | {count/max(time.monotonic()-started,.001):.1f}/s",flush=True)
                if count>=a.limit:break
        out.flush()
    if count<100:raise RuntimeError(f"UltraChat yielded only {count} turns; partial retained")
    partial.replace(a.output); a.output.with_suffix(".manifest.json").write_text(json.dumps({"dataset":DATASET,"split":SPLIT,"examples":count,"source_url":"https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k","license_declared_by_dataset_card":"MIT","warning":"Generated conversations are not independently fact-checked."},indent=2)+"\n",encoding="utf-8")
    print(f"[UltraChat] COMPLETE {count:,} turns",flush=True)
if __name__=="__main__":main()
