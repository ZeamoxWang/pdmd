"""CPU-only token count and cache-size estimate; does not rewrite text."""
import argparse
import json
import time
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prompts',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    tokenizer=Tokenizer.from_file(a.tokenizer)
    count=total=maximum=0
    started=time.monotonic()
    batch=[]
    def consume(batch):
        lengths=[len(e.ids) for e in tokenizer.encode_batch(batch,add_special_tokens=False)]
        return len(lengths),sum(lengths),max(lengths,default=0)
    with open(a.prompts) as f:
        for line in f:
            batch.append(json.loads(line)['prompt'])
            if len(batch)==512:
                n,t,m=consume(batch)
                count+=n; total+=t; maximum=max(maximum,m)
                batch=[]
    if batch:
        n,t,m=consume(batch)
        count+=n; total+=t; maximum=max(maximum,m)
    report={'records':count,'total_tokens':total,'maximum_tokens':maximum,
            'mean_tokens':total/count,'expected_feature_and_tag_gib':total*(5120*2+8)/2**30,
            'seconds':time.monotonic()-started,'additional_rewrite':False}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
