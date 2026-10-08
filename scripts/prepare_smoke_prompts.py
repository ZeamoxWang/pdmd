"""Eight untouched raw VidProM prompts for engineering checks, NOT paper training."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import urllib.request

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',default='data/smoke-original-8.jsonl')
out=Path(ap.parse_args().output)
out.parent.mkdir(parents=True,exist_ok=True)
revision='0c0900e6a037b058b1903bc1de9bcb6037d9fc24'
url=f'https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/{revision}/VidProM_unique.csv'
with urllib.request.urlopen(url) as response:
    reader=csv.DictReader(io.TextIOWrapper(response,encoding='utf-8',newline=''))
    rows=[{'id':r['uuid'],'prompt':r['prompt']} for r in itertools.islice(reader,8)]
if len(rows)!=8:raise ValueError('Expected 8 complete original rows')
text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
if out.exists() and out.read_text()!=text:raise ValueError('Existing smoke source differs')
out.write_text(text)
out.with_suffix('.meta.json').write_text(json.dumps({
    'source_url':url,'revision':revision,'selection':'first 8 rows, no changes',
    'purpose':'engineering smoke test, NOT the 248K paper subset',
    'sha256':hashlib.sha256(text.encode()).hexdigest()},indent=2)+'\n')
print(out,flush=True)
