"""Reusable exact-parameter preflight. Produces a receipt; never runs diffusion."""
import ast,copy,json,pickle,warnings,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import scripts.preflight_candidate1583 as prior
from driveloop.native_phase import NativePhaseRegistry,digest
from driveloop.phase_input_receipt import PhaseInputReceipt
from scripts.run_native_bus_control import ROOT,setup_env,make_backend,REQUEST,assert_native
from scripts.dd2_capture_hooks import PIPE,TESTER

OUT=ROOT/'outputs/revision_20261014_exact_phase_input'
RANGE=ROOT/'outputs/revision_20261013_phase_range'
PRE=ROOT/'outputs/revision_20261008_candidate1583_inputs/preflight.json'
PRE_SHA='997384cdb119b1679abc84e48276da54f38e3141d297271e50e192bd499d9af2'
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def validate_phase_parameter(registry, gm_path, out, phase, provenance_files=()):
 OUT=Path(out);OUT.mkdir(parents=True,exist_ok=False);prior.torch.set_num_threads(2);start=time.monotonic()
 gm_path=Path(gm_path);gm=json.loads(gm_path.read_text())
 assert gm['geometry']['passed'] and gm['selected_delay_seconds']==phase
 native=Path(registry.case['inputs']['original']['dataset']);edited=Path(gm['edited_dataset'])
 assert native.resolve()==Path(gm['native_dataset']).resolve() and gm['instance_token']==registry.instance
 assert digest(native/'labels/data.pkl')==gm['native_labels_sha256'] and digest(edited/'labels/data.pkl')==gm['edited_labels_sha256']
 input_manifest={**registry.case['inputs']['original'],'dataset':str(edited)}
 files=list(map(Path,provenance_files))+[gm_path,gm_path.parent/'geometry_checks.json',gm_path.parent/'selection_protocol.json',edited/'labels/data.pkl',edited/'config.json',edited/'labels/config.json',Path(__file__).resolve(),ROOT/'driveloop/phase_input_receipt.py',ROOT/'scripts/preflight_candidate1583.py']
 for part in ('images','hdmaps'):
  assert (edited/part).resolve()==(native/part).resolve()
  files.extend([native/part/'data.mdb',native/part/'config.json'])
 fixed={str(p):digest(p) for p in files}
 save(OUT/'protocol.json',{'phase_s':phase,'origin':'Exact parameter requested by dynamic admission; provenance pinned separately','generation_run':False,'source_registry_sha256':registry.fingerprint,'input_manifest':input_manifest,'fixed_hashes':fixed,'script_sha256':digest(__file__),'checks':['pose-only label edit','all48 view/frame transforms','bus channel only','all6 first frames unchanged','cached text/image/map unchanged','actual baseline and proposal loaders','same scene and instance'],'scope':'One exact parameter only; no neighboring or continuous-interval authorization.'})
 records=pickle.loads((native/'labels/data.pkl').read_bytes());updated=pickle.loads((edited/'labels/data.pkl').read_bytes());assert len(records)==len(updated)==144
 changes=[]
 for idx,(a,b) in enumerate(zip(records,updated)):
  assert a.keys()==b.keys()
  for key in a:
   if key!='boxes3d':assert pickle.dumps(a[key],protocol=4)==pickle.dumps(b[key],protocol=4),(idx,key)
  assert a['boxes3d'].shape==b['boxes3d'].shape
  for j in np.flatnonzero(np.any(a['boxes3d']!=b['boxes3d'],axis=1)):
   assert a['instance_tokens'][j]==registry.instance and idx%24 in range(3,22,3)
   assert np.array_equal(a['boxes3d'][j,3:7],b['boxes3d'][j,3:7]) and a['boxes3d'][j,8]==b['boxes3d'][j,8]
   changes.append({'record_index':idx,'instance_token':registry.instance})
 for cam in range(6):assert np.array_equal(records[cam*24]['boxes3d'],updated[cam*24]['boxes3d'])
 assert changes
 transform_path=ROOT/'dreamer-train/projects/DriveDreamer2/drivedreamer2/drivedreamer2_transforms.py'
 cls=next(n for n in ast.parse(transform_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DriveDreamer2_Transform')
 ns=dict(vars(prior),HEIGHT=900,WIDTH=1600)
 exec(compile(ast.Module(body=[cls],type_ignores=[]),str(transform_path),'exec'),ns)
 transform=ns['DriveDreamer2_Transform'](resolution=(256,448),gd_input_name='image_hdmap',bd_input_name='image_box',is_train=False,embed_map_path='/data/projects/DriveLoop/clip_text_transform_after_pool_panoramic.pkl',text_embed_dim=1024,box_normal=True)
 dataset=prior.load_dataset(str(native));raster=[]
 for cam in range(6):
  for frame in range(8):
   idx=cam*24+frame*3;a=dataset[idx];b=copy.deepcopy(a);b['boxes3d']=updated[idx]['boxes3d'].copy()
   with warnings.catch_warnings():
    warnings.simplefilter('error',RuntimeWarning);aa=transform(a);bb=transform(b)
   x=aa['box_downsampler_input'].numpy();y=bb['box_downsampler_input'].numpy()
   channels=[c for c in range(19) if not np.array_equal(x[c],y[c])]
   assert set(channels)<={11}
   if frame==0:assert not channels
   for key in ('grounding_downsampler_input','input_image','prompt_embeds'):assert np.array_equal(aa[key],bb[key])
   raster.append({'camera':cam,'frame':frame,'changed_channels':channels,'changed_pixels':int(np.count_nonzero(x!=y))})
 assert any(x['changed_channels'] for x in raster);del dataset
 audits={};generations={}
 for name,manifest in [('original',registry.case['inputs']['original']),('proposal',input_manifest)]:
  dest=OUT/name;previous_capture=os.environ.get('DRIVELOOP_CAPTURE_DIR');os.environ['DRIVELOOP_CAPTURE_DIR']=str(dest/'capture')
  backend=make_backend(manifest,dest,audit_only=True);backend.python_executable=str(ROOT/'scripts/dd2_capture_python')
  try:g=backend.generate(REQUEST,0)
  finally:
   if previous_capture is None:os.environ.pop('DRIVELOOP_CAPTURE_DIR',None)
   else:os.environ['DRIVELOOP_CAPTURE_DIR']=previous_capture
  assert_native(g,manifest)
  hook=json.loads((dest/'capture/hook_state.json').read_text())
  assert set(hook['installed'])=={PIPE,TESTER} and hook['counts']=={'decode':0,'postprocess':0,'composite':0} and not hook['files']
  assert g.metadata['dd2_audit_only'] and 'video' not in g.artifacts
  selector=g.metadata['dd2_source_sample_binding']['selector']
  assert selector['scene_token']==registry.scene and selector['instance_token']==registry.instance
  save(dest/'generation.json',asdict(g));generations[name]=g;audits[name]=g.metadata['dd2_runtime_input_audit']
  print('EXACT_PHASE_LOADER_PASS',name,flush=True)
 for key in ('prompt_embed','img_cond','grounding_downsampler_input','box_downsampler_input'):
  assert audits['original'][key]['sha256']==registry.case['runtime_audits']['original'][key]['sha256']
 for key in ('prompt_embed','img_cond','grounding_downsampler_input'):
  assert audits['proposal'][key]['sha256']==audits['original'][key]['sha256']
 assert audits['proposal']['box_downsampler_input']['sha256']!=audits['original']['box_downsampler_input']['sha256']
 for p,h in fixed.items():assert digest(p)==h,p
 registry.verify()
 for name in ('original','proposal'):
  for p in (OUT/name).rglob('*.json'):fixed[str(p)]=digest(p)
 receipt={'schema':'exact_phase_input_v1','passed':True,'generation_run':False,'phase_s':phase,'registry_sha256':registry.fingerprint,
  'scene_token':registry.scene,'instance_token':registry.instance,'candidate_index':registry.candidate,
  'geometry_manifest':str(gm_path),'input_manifest':input_manifest,'fixed_hashes':fixed,
  'geometry_passed':True,'raster_passed':True,'runtime_passed':True,'first_frames_unchanged':True,
  'runtime_audit':audits['proposal'],'baseline_runtime_audit':audits['original'],'changed_records':changes,'raster_audit':raster,
  'actual_loader_calls':2,'diffusion_calls':0,'scope':'Exact phase and source only; not video quality, physical safety or general range validation.'}
 save(OUT/'receipt.json',receipt)
 sealed=PhaseInputReceipt(OUT/'receipt.json',digest(OUT/'receipt.json'),phase,registry)
 assert sealed.verify_runtime(generations['proposal'])
 save(OUT/'summary.json',{'passed':True,'generation_run':False,'phase_s':phase,'receipt_sha256':digest(OUT/'receipt.json'),
  'labels_changed_records':len(changes),'raster_frames_checked':48,'raster_frames_changed':sum(bool(x['changed_channels']) for x in raster),
  'only_changed_channel':11,'all_six_first_frames_unchanged':True,'actual_loader_calls':2,'diffusion_calls':0,
  'receipt_reload_and_runtime_match':True,'elapsed_seconds':time.monotonic()-start})
 print('EXACT_PHASE_INPUT_PREFLIGHT_COMPLETE',OUT,flush=True)
 return sealed
