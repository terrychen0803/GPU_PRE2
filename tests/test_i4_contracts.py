import unittest
from gpu_pre2.data.contracts import (validate_row,predictor_view,
                                      validate_predictor_view,audit_splits)
def row(kind="isolated",group="train-a",split="train"):
    n=2 if kind=="isolated" else 3
    gold=([{"start":0,"end":2,"class_id":0}]
          if kind=="isolated" else
          [{"start":0,"end":2,"class_id":0},
           {"start":2,"end":3,"class_id":1}])
    return {"sample_id":group+"-"+kind,"group_id":group,
            "source_gpu":"RTX4090","split":split,"kind":kind,
            "source_event_features":[[0.01]*27 for _ in range(n)],
            "source_hw":[0.5,0.3],"gold_segments":gold,
            "quality":{"usable_for_segmentation":True}}
class Contracts(unittest.TestCase):
    def test_safe_inference_view(self):
        r=row()
        validate_row(r)
        v=predictor_view(r)
        self.assertEqual(set(v),{"source_event_features","source_hw"})
        with self.assertRaises(ValueError):
            validate_predictor_view({**v,"gold_segments":r["gold_segments"]})
    def test_group_leak(self):
        a=row()
        b=row("context_d","train-a","validation")
        with self.assertRaisesRegex(ValueError,"group leakage"):
            audit_splits([a,b])
    def test_bad_boundary(self):
        r=row()
        r["gold_segments"][0]["end"]=1
        with self.assertRaises(ValueError):
            validate_row(r)
    def test_unsupported_class(self):
        r=row()
        r["gold_segments"][0]["class_id"]=11
        with self.assertRaises(ValueError):
            validate_row(r)
if __name__=="__main__":unittest.main()
