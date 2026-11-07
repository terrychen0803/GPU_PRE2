import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gpu_pre2.data import storage
from gpu_pre2.i3.context import assets as context_assets
from gpu_pre2.i3.context.protocol import protocol as context_protocol
from gpu_pre2.profiling.protocols import validate_request, GROUPS4


class RetainedIsolation(unittest.TestCase):
    def test_runtime_has_no_old_imports_or_path_injection(self):
        forbidden={'o1_operator','o2_sequence','o4_iteration','o4_research','inference_pilot','o4_i3','o4_i3_v2','o4_i3_context'}
        for p in (storage.REPO/'src').rglob('*.py'):
            text=p.read_text()
            self.assertNotIn('sys.path.insert',text,str(p))
            self.assertNotIn('sys.path.append',text,str(p))
            for node in ast.walk(ast.parse(text)):
                if isinstance(node,ast.ImportFrom) and node.level==0:
                    self.assertNotIn((node.module or '').split('.')[0],forbidden,str(p))
                if isinstance(node,ast.Import):
                    self.assertFalse(forbidden.intersection(x.name.split('.')[0] for x in node.names),str(p))

    def test_old_stages_and_unknown_dataset_ids_rejected(self):
        with self.assertRaises(ValueError):storage.dataset('old_O2',stage='O2')
        with self.assertRaises(ValueError):storage.dataset('auto_discover_latest',stage='I3')

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'retained';root.mkdir()
            (root/'old').symlink_to(Path(folder))
            with patch.dict('os.environ',{'GPU_PRE2_DATA_ROOT':str(root)}):
                with self.assertRaises(ValueError):storage.inside_data(root/'old/outside.json')

    def test_removed_protocol_and_mislabeled_groups4_rejected(self):
        with self.assertRaises(ValueError):validate_request({'protocol':{'protocol_id':'o2_formal'}})
        with self.assertRaises(ValueError):validate_request({'protocol':{'protocol_id':GROUPS4,'single_stream_required':False},'config_id':'other'})

    def test_both_context_versions_use_shipped_plans(self):
        try:
            for plan_id,count in [('A',14),('BC',26)]:
                context_assets.select_plan(plan_id)
                self.assertEqual(len(context_assets.plan()['work_items']),count)
                p=context_protocol('formal')
                self.assertTrue(p['protocol_id'].startswith('gpu_pre2:'))
                self.assertTrue(p['source_manifest_required'])
                self.assertFalse(p['source_snapshot_required'])
        finally:context_assets.select_plan('BC')

    def test_imported_collectors_resolve_inside_new_repo(self):
        modules=['gpu_pre2.i2.run_formal','gpu_pre2.i3.isolated.collect',
                 'gpu_pre2.i3.context.collect','gpu_pre2.models.historical.predict']
        for name in modules:
            module=importlib.import_module(name)
            self.assertIn(storage.REPO,Path(module.__file__).resolve().parents)
