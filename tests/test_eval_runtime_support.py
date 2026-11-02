import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from driveloop.schema import DriveLoopRequest, Generation, Evaluation, DriveLoopConfig
from driveloop.runner import DriveLoopRunner
from scripts.eval_runtime_support import SlotBackend, InterruptedSlot, completion

class SupportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.calls=[]
    def tearDown(self):self.temp.cleanup()
    def factory(self,dest):
        def generate(request,iteration):
            self.calls.append((dest,iteration));p=dest/'video.mock';p.write_bytes(b'NOT_A_REAL_VIDEO')
            return Generation(iteration,request.prompt,{'video':str(p)},{'mock_test_only':True})
        return NS(generate=generate)
    def backend(self,**kw):return SlotBackend(self.root,'frozen-config',self.factory,**kw)
    def test_resume_reuses_and_request_drift_fails(self):
        req=DriveLoopRequest('original')
        a=self.backend().generate(req,0);b=self.backend().generate(req,0)
        self.assertEqual(a,b);self.assertEqual(len(self.calls),1)
        with self.assertRaisesRegex(ValueError,'changed'):self.backend().generate(DriveLoopRequest('changed'),0)
    def test_artifact_tamper_is_not_regenerated(self):
        req=DriveLoopRequest('original');g=self.backend().generate(req,0)
        Path(g.artifacts['video']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.backend().generate(req,0)
        self.assertEqual(len(self.calls),1)
    def test_receipt_tamper(self):
        req=DriveLoopRequest('original');self.backend().generate(req,0)
        p=self.root/'slot_00/receipt.json';x=json.loads(p.read_text());x['payload']['generation']['prompt']='changed';p.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'integrity'):self.backend().generate(req,0)
    def test_interrupted_slot_requires_explicit_retry_preserves_evidence(self):
        req=DriveLoopRequest('original')
        def broken(dest):return NS(generate=lambda r,i:(_ for _ in ()).throw(RuntimeError('simulated interruption')))
        with self.assertRaises(RuntimeError):SlotBackend(self.root,'frozen-config',broken).generate(req,0)
        with self.assertRaises(InterruptedSlot):self.backend().generate(req,0)
        self.backend(retry_incomplete=True).generate(req,0)
        self.assertTrue((self.root/'slot_00/try_000/failed.json').exists())
        self.assertTrue((self.root/'slot_00/try_001/started.json').exists())
        self.assertEqual(self.calls[0][1],0)
    def result(self,n,flags=(),returned='budget_exhausted'):
        return NS(attempt_history=[NS(iteration=i,evaluation=Evaluation(0,{'perception_view_restriction_unresolved':i in flags})) for i in range(n)],return_status=returned,returned_generation=NS(iteration=0))
    def test_early_stop_not_execution_error(self):
        self.assertEqual(completion(self.result(1,[0]))['termination'],'perception_unmeasurable')
        self.assertEqual(completion(self.result(2,[1],'accepted'))['return_status'],'accepted')
    def test_truncation_and_continuing_after_stop_fail(self):
        with self.assertRaises(ValueError):completion(self.result(2))
        with self.assertRaises(ValueError):completion(self.result(4,[0]))
        self.assertEqual(completion(self.result(4))['termination'],'candidate_cap_reached')
    def test_real_runner_stops_and_replays_one_unmeasurable_attempt(self):
        evaluator=NS(evaluate=lambda g:Evaluation(0,{'perception_view_restriction_unresolved':True}))
        req=DriveLoopRequest('urban street, a bus cuts in from the left toward the ego vehicle')
        for n in range(2):
            runner=DriveLoopRunner(self.backend(),evaluator=evaluator,
                config=DriveLoopConfig(max_iterations=4,use_task_utility=True,
                    acceptance_policy='joint_perception_direction',stop_on_acceptance=False,
                    output_dir=self.root/f'history{n}'))
            result=runner.run(req)
            self.assertEqual(completion(result)['termination'],'perception_unmeasurable')
            self.assertEqual(result.return_status,'perception_unmeasurable')
        self.assertEqual(len(self.calls),1)

if __name__=='__main__':unittest.main()
