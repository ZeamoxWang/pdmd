"""Cache acceptance rejects changed prompt identity and nonfinite features."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from scripts.verify_cache import verify


class CacheAuditTest(unittest.TestCase):
    def test_original_identity_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root/'prompts.jsonl'
            prompts.write_text(json.dumps({'id': 'raw-17', 'prompt': 'Two birds.'})+'\n')
            sha = hashlib.sha256(prompts.read_bytes()).hexdigest()
            row = {'id': 'raw-17', 'source_row': 0, 'key': 'sample_00000000',
                   'tokens': 2, 'world_size': 1, 'shard': 'features.safetensors',
                   'source_sha256': sha,
                   'prompt_sha256': hashlib.sha256(b'Two birds.').hexdigest()}
            metadata = {'source_sha256': sha, 'encoder_layer': '50', 'final_norm': 'identity',
                        'prompt_rewrite': 'false',
                        'base_revision': '42ed227ee7df40d41602854ae760620d6eb651fe',
                        'diffsynth_revision': '974cfa37f27ac55eba3b6d10efa21f876900572d'}
            (root/'COMPLETE-rank-0000.json').write_text(json.dumps({
                'source_sha256': sha, 'world_size': 1, 'limit': None}))
            manifest = root/'rank-0000.jsonl'
            manifest.write_text(json.dumps(row)+'\n')
            features = {'sample_00000000': torch.ones(2, 5120, dtype=torch.bfloat16),
                        'sample_00000000_tags': torch.ones(2, dtype=torch.long)}
            save_file(features, root/'features.safetensors', metadata=metadata)
            self.assertEqual(verify(root, prompts, 1)['samples'], 1)
            row['prompt_sha256'] = hashlib.sha256(b'A rewritten prompt').hexdigest()
            manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'prompt identity'):
                verify(root, prompts, 1)
            row['prompt_sha256'] = hashlib.sha256(b'Two birds.').hexdigest()
            manifest.write_text(json.dumps(row)+'\n')
            features['sample_00000000'][0, 0] = float('nan')
            save_file(features, root/'features.safetensors', metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'embedding values'):
                verify(root, prompts, 1)
            features['sample_00000000'][0, 0] = 1
            features['sample_00000000_tags'][0] = 0
            save_file(features, root/'features.safetensors', metadata=metadata)
            with self.assertRaisesRegex(ValueError, 'token tags'):
                verify(root, prompts, 1)


if __name__ == '__main__':
    unittest.main()
