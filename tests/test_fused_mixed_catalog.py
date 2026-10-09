import json,tempfile,unittest
from pathlib import Path
from mlir_oot.fused_mixed_catalog import sha,stage_capture

class TestFusedMixedCatalog(unittest.TestCase):
    def create(self,root):
        capture=root/'source';bundle=root/'bundle';capture.mkdir();bundle.mkdir()
        for name in ['model.mlir','weights.safetensors','weights.safetensors.manifest.json']:(capture/name).write_text(name)
        (capture/'capture_receipt.json').write_text(json.dumps({'artifacts':{}}))
        (bundle/'rewritten.mlir').write_text('rewritten');(bundle/'requant.o').write_bytes(b'object')
        pairs=[(capture/'model.mlir','source_sha256'),(capture/'weights.safetensors','weights_sha256'),(capture/'weights.safetensors.manifest.json','manifest_sha256'),(bundle/'rewritten.mlir','rewritten_sha256'),(bundle/'requant.o','object_sha256')]
        (bundle/'requant.json').write_text(json.dumps({key:sha(path) for path,key in pairs}))
        return capture,bundle

    def test_derived_capture_preserves_identity_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);capture,bundle=self.create(root);stage=stage_capture(capture,bundle,root/'derived')
            self.assertEqual((stage/'model.mlir').read_text(),'rewritten')
            self.assertEqual(sha(stage/'weights.safetensors'),sha(capture/'weights.safetensors'))
            receipt=json.loads((stage/'capture_receipt.json').read_text())
            self.assertEqual(receipt['derived_from']['source_sha256'],sha(capture/'model.mlir'))
            self.assertEqual(receipt['artifacts']['model.mlir']['sha256'],sha(stage/'model.mlir'))

    def test_altered_weights_refused_before_staging(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);capture,bundle=self.create(root);(capture/'weights.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):stage_capture(capture,bundle,root/'derived')
            self.assertFalse((root/'derived').exists())
