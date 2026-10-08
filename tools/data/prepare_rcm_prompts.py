"""Fetch rCM's publicly linked prompt list without rewriting any prompt text."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = '2f8b779212da279d212c22a509b66ad6552f350e'
URL = ('https://huggingface.co/gdhe17/Self-Forcing/resolve/' + REVISION +
       '/vidprom_filtered_extended.txt')
SOURCE_SHA256 = '7896742f468bc8aef9e4547424d1ce0a951acdb2a82233790155401a99bf5aa5'
RCM_COMMIT = 'ed3cb14dd936f92cdc9f9381af7369991509b41f'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    source, output, receipt = root/'source.txt', root/'prompts.jsonl', root/'provenance.json'
    if not source.exists():
        temp = source.with_suffix('.txt.partial')
        with urllib.request.urlopen(URL, timeout=120) as response, temp.open('wb') as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        if digest(temp) != SOURCE_SHA256:
            raise ValueError('Downloaded public source failed SHA-256 verification')
        temp.rename(source)
    if digest(source) != SOURCE_SHA256:
        raise ValueError('Existing source differs from the pinned public file')
    if output.exists():
        data = json.loads(receipt.read_text())
        if data['source_sha256'] != SOURCE_SHA256 or data['output_sha256'] != digest(output):
            raise ValueError('Existing prepared prompts failed provenance verification')
        print(json.dumps(data), flush=True)
        return
    temp = output.with_suffix('.jsonl.partial')
    count, max_chars = 0, 0
    with source.open(encoding='utf-8', newline='') as f, temp.open('w', encoding='utf-8') as g:
        for index, line in enumerate(f):
            prompt = line.removesuffix('\n').removesuffix('\r')
            if not prompt.strip():
                raise ValueError(f'Blank source row {index}; no implicit dropping')
            g.write(json.dumps({'id': str(index), 'prompt': prompt}, ensure_ascii=False)+'\n')
            count += 1
            max_chars = max(max_chars, len(prompt))
    if count == 0:
        raise ValueError('Empty corpus')
    data = {'source_url': URL, 'source_revision': REVISION, 'source_sha256': SOURCE_SHA256,
            'rcm_reference': f'https://github.com/NVlabs/rcm/blob/{RCM_COMMIT}/README.md#dataset-downloading',
            'records': count, 'max_characters': max_chars, 'output_sha256': digest(temp),
            'output': str(output), 'additional_rewrite': False, 'filtering': False,
            'text_preservation': 'Only physical line separators removed; source spaces preserved',
            'source_text_status': 'Published rCM-linked extended text, used verbatim'}
    # Receipt first: a retry never mistakes an incomplete output for a verified file.
    receipt.write_text(json.dumps(data, indent=2)+'\n')
    temp.rename(output)
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
