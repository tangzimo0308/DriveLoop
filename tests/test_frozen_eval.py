"""CPU-only failure/recovery tests. Mock outputs are never experimental data."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from dataclasses import asdict
from scripts import run_frozen_eval as m
from scripts.eval_runtime_support import SlotBackend, PresetRefiner, sha, digest, completion
from driveloop.schema import DriveLoopRequest, Generation, Evaluation, Diagnosis, DriveLoopConfig
from driveloop.runner import DriveLoopRunner
from driveloop.refiner import RuleBasedRefiner

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def test_pair_checks_actual_noise_not_configured_seed(self):
        for method in m.METHODS:m.pair_noise(self.root,method,0,{'sha256':'same'})
        r=m.unseal(self.root/'noise_pairs/00.json');self.assertTrue(r['all_four_observed'])
        with self.assertRaisesRegex(ValueError,'noise mismatch'):m.pair_noise(self.root,'feedback',0,{'sha256':'different'})
        self.assertEqual(m.unseal(self.root/'noise_pairs/00.json'),r)
    def test_partial_pair_not_claimed_complete(self):
        m.pair_noise(self.root,'resample',0,{'sha256':'same'})
        self.assertFalse(m.unseal(self.root/'noise_pairs/00.json')['all_four_observed'])
    def test_envelope_and_file_tamper(self):
        p=self.root/'a.json';m.seal(p,{'value':1});e=m.read(p);e['payload']['value']=2;m.atomic_json(p,e)
        with self.assertRaises(ValueError):m.unseal(p)
        p.write_text('a');files={str(p):sha(p)};p.write_text('b')
        with self.assertRaises(ValueError):m.verify_files(files)
    def test_effective_request_drift_rejected_before_call(self):
        called=[];backend=NS(generate=lambda r,i:called.append(i));request=DriveLoopRequest('first')
        checked=m.CheckedBackend(backend,{}, {0:digest(asdict(request))})
        checked.generate(request,0)
        with self.assertRaises(ValueError):checked.generate(DriveLoopRequest('changed'),0)
        self.assertEqual(called,[0])
    def test_frozen_file_change_during_run(self):
        p=self.root/'weights';p.write_text('a');stats=m.stat_snapshot({str(p):sha(p)})
        checked=m.CheckedBackend(NS(generate=lambda r,i:None),stats,None);p.write_text('changed')
        with self.assertRaises(ValueError):checked.generate(DriveLoopRequest('x'),0)
    def test_recovery_real_runner_feedback_same_requests_no_extra_generation(self):
        calls=[]
        def factory(dest):
            def generate(r,i):
                calls.append(i);p=dest/'fake';p.write_text('TEST_ONLY')
                return Generation(i,r.prompt,{'test_only':str(p)},{'mock_test_only':True})
            return NS(generate=generate)
        req=DriveLoopRequest('urban street, a bus cuts in from the left toward the ego vehicle',condition={'generation_escalation':{'num_inf_steps':50}})
        for replay in range(2):
            def evaluate(g):
                if replay==0 and g.iteration==1:raise RuntimeError('simulated evaluator interruption')
                return Evaluation(.2,{'maneuver_direction_consistent':False},Diagnosis(False,['target_object_not_detected','low_detection_coverage']))
            runner=DriveLoopRunner(SlotBackend(self.root/'slots','same',factory),NS(evaluate=evaluate),RuleBasedRefiner(),
                config=DriveLoopConfig(max_iterations=4,output_dir=self.root/f'history{replay}',use_task_utility=True,
                    acceptance_policy='joint_perception_direction',stop_on_acceptance=False))
            if replay==0:
                with self.assertRaisesRegex(RuntimeError,'interruption'):runner.run(req)
            else:self.assertEqual(completion(runner.run(req))['attempt_count'],4)
        self.assertEqual(calls,[0,1,2,3])
    def test_failed_evaluation_still_charged(self):
        evaluator=NS(detector=NS(get_runtime_audit=lambda:{'successful_predict_calls':3}),
            evaluate=lambda g:(_ for _ in ()).throw(RuntimeError('fail')))
        m.instrument_evaluator(evaluator,self.root)
        with self.assertRaises(RuntimeError):evaluator.evaluate(NS(iteration=0))
        row=m.read(self.root/'evaluation_calls/00.json');self.assertFalse(row['complete']);self.assertEqual(row['detector_runtime']['successful_predict_calls'],3)
    def test_changed_evaluation_on_replay_fails(self):
        evaluator=NS(detector=NS(get_runtime_audit=lambda:{'successful_predict_calls':3}),evaluate=lambda g:Evaluation(.4))
        m.instrument_evaluator(evaluator,self.root,{0:asdict(Evaluation(.2))})
        with self.assertRaisesRegex(ValueError,'changed on generation replay'):evaluator.evaluate(NS(iteration=0))
        self.assertFalse(m.read(self.root/'evaluation_calls/00.json')['complete'])
    def test_ledger_retains_all_assignments_and_retry_cost(self):
        group=self.root/'groups/case/11'
        for n in (0,1):m.atomic_json(group/f'fixed/slots/slot_00/try_{n:03d}/started.json',{})
        f={'assigned_arms':[{'case_id':'case','bank':11,'method':method} for method in m.METHODS]}
        with patch.object(m,'OUT',self.root):r=m.ledger(f)
        self.assertEqual(len(r['rows']),4);self.assertEqual(r['complete_arms'],0)
        self.assertEqual(r['generation_calls_started'],2);self.assertEqual(r['extra_retry_calls_started'],1)

if __name__=='__main__':unittest.main()
