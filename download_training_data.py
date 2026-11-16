"""Resumable, progress-reporting CPU data downloads for FirstAI."""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path

def download_wikitext(output, config, max_chars, progress_every=100_000):
    from datasets import load_dataset
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(target.suffix+".partial")
    if target.exists() and target.stat().st_size:
        print(f"[WikiText] already complete: {target.stat().st_size:,} bytes",flush=True); return target
    saved_rows=0; saved_chars=0
    if partial.exists():
        with partial.open(encoding="utf-8") as f:
            for line in f:
                saved_rows+=1; saved_chars+=len(line)
        print(f"[WikiText] RESUME: {saved_rows:,} rows, {saved_chars:,} chars saved",flush=True)
    print(f"[WikiText] opening streaming dataset {config}",flush=True)
    ds=load_dataset("Salesforce/wikitext",config,split="train",streaming=True)
    started=time.monotonic(); chars=saved_chars; last=chars
    with partial.open("a" if saved_rows else "w",encoding="utf-8") as out:
        for idx,row in enumerate(ds):
            if idx<saved_rows: continue
            text=str(row.get("text") or "").strip()
            if not text: continue
            chunk=text+"\n"
            if chars+len(chunk)>max_chars: chunk=chunk[:max_chars-chars]
            if not chunk: break
            out.write(chunk); chars+=len(chunk)
            if chars-last>=progress_every:
                out.flush()
                elapsed=max(time.monotonic()-started,.001)
                print(f"[WikiText] {chars:,}/{max_chars:,} chars ({chars/max_chars:.1%}) | source row {idx:,} | {((chars-saved_chars)/elapsed):,.0f} chars/s",flush=True)
                last=chars
            if chars>=max_chars: break
        out.flush()
    if chars<20_000: raise RuntimeError(f"WikiText only produced {chars:,} chars; partial kept at {partial}")
    partial.replace(target)
    manifest={"dataset":"Salesforce/wikitext","config":config,"split":"train","source_url":"https://huggingface.co/datasets/Salesforce/wikitext","retrieved_at_utc":datetime.now(timezone.utc).isoformat(),"characters":chars,"target_characters":max_chars}
    target.with_suffix(target.suffix+".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"[WikiText] COMPLETE: {chars:,} chars -> {target}",flush=True)
    return target

def download_dolly(output,max_examples=15000,extra_examples=None,progress_every=500):
    from datasets import load_dataset
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(target.suffix+".partial")
    if target.exists() and target.stat().st_size:
        print(f"[Dolly] already complete: {target}",flush=True); return target
    count=0
    if partial.exists():
        with partial.open(encoding="utf-8") as f: count=sum(1 for x in f if x.strip())
    print(f"[Dolly] streaming databricks-dolly-15k; resume at {count:,}",flush=True)
    ds=load_dataset("databricks/databricks-dolly-15k",split="train",streaming=True)
    started=time.monotonic()
    with partial.open("a" if count else "w",encoding="utf-8") as out:
        for idx,row in enumerate(ds):
            if idx<count: continue
            if count>=max_examples: break
            instruction=str(row.get("instruction") or "").strip()
            response=str(row.get("response") or "").strip()
            context=str(row.get("context") or "").strip()
            if not instruction or not response: continue
            prompt=instruction+(("\n\nReference context:\n"+context) if context else "")
            item={"prompt":prompt,"response":response,"metadata":{"dataset_id":"dolly15k","dataset":"databricks/databricks-dolly-15k","split":"train","license":"CC-BY-SA-3.0","source_url":"https://huggingface.co/datasets/databricks/databricks-dolly-15k","source_record_index":idx}}
            out.write(json.dumps(item,ensure_ascii=False)+"\n"); count+=1
            if count%progress_every==0:
                out.flush(); elapsed=max(time.monotonic()-started,.001)
                print(f"[Dolly] {count:,}/{max_examples:,} examples | {count/elapsed:.1f} examples/s",flush=True)
        out.flush()
    if extra_examples and Path(extra_examples).is_file():
        with partial.open("a",encoding="utf-8") as out,Path(extra_examples).open(encoding="utf-8") as src:
            for line in src:
                if line.strip(): out.write(line); count+=1
    if count<4: raise RuntimeError(f"Dolly produced {count} examples; partial retained")
    partial.replace(target)
    target.with_suffix(target.suffix+".manifest.json").write_text(json.dumps({"dataset":"databricks/databricks-dolly-15k","examples_including_local_extras":count,"source_url":"https://huggingface.co/datasets/databricks/databricks-dolly-15k"},indent=2)+"\n",encoding="utf-8")
    print(f"[Dolly] COMPLETE: {count:,} examples",flush=True)
    return target

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wikitext-out",default="data/raw/wikitext_train.txt")
    p.add_argument("--wikitext-config",choices=["wikitext-2-raw-v1","wikitext-103-raw-v1"],default="wikitext-103-raw-v1")
    p.add_argument("--max-chars",type=int,default=30_000_000)
    p.add_argument("--instructions-out",default="data/orbit_instructions_full.jsonl")
    p.add_argument("--max-instructions",type=int,default=15000)
    p.add_argument("--extra-instructions",default="data/orbit_instructions.example.jsonl")
    p.add_argument("--skip-wikitext",action="store_true"); p.add_argument("--skip-instructions",action="store_true")
    a=p.parse_args()
    if not a.skip_wikitext: download_wikitext(a.wikitext_out,a.wikitext_config,a.max_chars)
    if not a.skip_instructions: download_dolly(a.instructions_out,a.max_instructions,a.extra_instructions)
if __name__=="__main__": main()
