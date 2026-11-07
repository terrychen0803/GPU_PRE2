import unittest
import torch
from gpu_pre2.models.semimarkov import (
    SemiMarkovRecognizer,structured_loss,log_partition,viterbi)
from gpu_pre2.models.n0 import N0TimeRatio,log_huber_loss
from gpu_pre2.models.evaluate import typed_f1,exact_boundary_f1
from gpu_pre2.models.train_mixed import fit_i4
from test_i4_contracts import row

class I4Models(unittest.TestCase):
    def test_dp_exact_small(self):
        # N=2, L=2, C=1: valid partitions [2] and [1,1].
        s=torch.tensor([[[0.5],[0.7]],[[0.3],[-float("inf")]]],
                       requires_grad=True)
        z=log_partition(s)
        expect=torch.logsumexp(torch.tensor([0.7,0.8]),dim=0)
        self.assertTrue(torch.allclose(z,expect,atol=1e-6))
        z.backward()
        self.assertIsNotNone(s.grad)
    def test_segment_training_and_viterbi(self):
        m=SemiMarkovRecognizer()
        x=torch.randn(5,27)
        scores=m(x,torch.tensor([0.1,0.2]))
        gold=[{"start":0,"end":2,"class_id":0},
              {"start":2,"end":5,"class_id":1}]
        loss=structured_loss(scores,gold)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        predicted=viterbi(scores)
        self.assertEqual(predicted[0]["start"],0)
        self.assertEqual(predicted[-1]["end"],5)
    def test_n0_head(self):
        m=N0TimeRatio()
        p=m(torch.zeros(3,27),torch.zeros(3,2),
            torch.ones(3,2),torch.ones(3)*10)
        self.assertEqual(tuple(p.shape),(3,))
        self.assertTrue(torch.all(p>0))
        self.assertTrue(torch.isfinite(log_huber_loss(p,torch.ones(3)*12)))
    def test_mixed_small(self):
        a=row("isolated","train-iso")
        b=row("context_d","train-ctx")
        c=row("context_i","val-ctx","validation")
        c["sample_id"]="val-context"
        model,norm,info=fit_i4([a,b],[c],mode="pretrain_mixed",seed=17,
                                pretrain_updates=1,second_updates=1,
                                log_every=1)
        self.assertEqual(info["mode"],"pretrain_mixed")
        self.assertEqual(len(info["history"]),2)
        self.assertEqual(len(norm.mean),27)
        self.assertIn("typed_f1",info["validation"])
if __name__=="__main__":unittest.main()
