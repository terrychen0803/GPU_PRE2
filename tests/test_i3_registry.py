import unittest
from gpu_pre2.i3.registry import catalog_summary
class I3Registry(unittest.TestCase):
    def test_core_catalog(self):
        info=catalog_summary()
        self.assertEqual(info["operator_catalog_count"],174)
        self.assertEqual(info["context_graph_count"],6)
        self.assertEqual(info["context_position_count"],28)
        self.assertEqual(info["context_work_items_theoretical"],40)
        self.assertEqual(info["context_leakage_groups"],["context_scale_0"])
if __name__=="__main__":
    unittest.main()
