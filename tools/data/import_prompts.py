"""Preserve raw prompt text in JSONL; never rewrite, deduplicate or filter silently."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--format',choices=['txt','vidprom-csv','jsonl'],required=True)
    p.add_argument('--limit',type=int,help='Smoke subset only; recorded in metadata')
    p.add_argument('--description',required=True,help='Exact origin and subset status')
    a=p.parse_args()
    source,out=Path(a.source),Path(a.output)
    if out.exists():raise FileExistsError(out)
    if a.limit is not None and a.limit<1:raise ValueError('limit must be positive')
    out.parent.mkdir(parents=True,exist_ok=True)
    count=0
    digest=hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    tmp=out.with_suffix(out.suffix+'.tmp')
    with source.open(newline='') as f,tmp.open('x') as g:
        if a.format=='vidprom-csv':
            rows=({'id':x['uuid'],'prompt':x['prompt']} for x in csv.DictReader(f))
        elif a.format=='jsonl':
            rows=(json.loads(x) for x in f)
        else:
            # Remove only the line separator, not spaces inside the original prompt.
            rows=({'id':str(i),'prompt':x.removesuffix('\n').removesuffix('\r')} for i,x in enumerate(f))
        for row in rows:
            if a.limit is not None and count>=a.limit:break
            if not isinstance(row['prompt'],str) or not row['prompt'].strip():
                raise ValueError(f'Blank/nontext prompt at row {count}; no implicit dropping')
            g.write(json.dumps({'id':str(row['id']),'prompt':row['prompt']},ensure_ascii=False)+'\n')
            count+=1
    if not count:raise ValueError('Empty source')
    tmp.rename(out)
    meta={'source':str(source),'source_sha256':digest.hexdigest(),'records':count,
          'description':a.description,'limit':a.limit,'rewritten':False,'filtered':False,
          'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(out.suffix+'.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__=='__main__':main()
