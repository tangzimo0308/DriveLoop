"""Frozen comparison policies for the amplitude development pilot."""
from driveloop.dynamic_phase import DynamicPhaseRefiner,action_id
from driveloop.schema import Refinement
METHODS=('direct_target','fixed_amplitude_search','residual_feedback')
class AmplitudePilotRefiner(DynamicPhaseRefiner):
    def __init__(self,admission,method,initial_phase):
        super().__init__(admission)
        if method not in METHODS or initial_phase not in (-.25,.25):raise ValueError('Unknown pilot policy')
        self.method,self.initial_phase=method,initial_phase
    def refine(self,request,evaluation):
        if self.method=='residual_feedback':return super().refine(request,evaluation)
        q=self.admission.contract(request);f=evaluation.metrics['native_phase_feedback']
        if f['contract']!=q:raise ValueError('Feedback contract mismatch')
        sign=1 if self.initial_phase>0 else -1
        sequence=[self.initial_phase,sign*.125,sign*.375]
        phase=self.initial_phase if self.method=='direct_target' else sequence[min(f['iteration']+1,2)]
        return Refinement(request.prompt,{'native_phase':{**q,'phase_s':phase,'arm':action_id(phase)}},['preset_policy:'+self.method])
