"""Verify copied source bytes and run the real DD2 input-audit path without generation."""
import copy,json,pickle,hashlib
from pathlib import Path
import lmdb,numpy as np
from dataclasses import asdict
from scripts.run_native_bus_control import ROOT,SOURCE,REQUEST,setup_env,make_backend,assert_native
from scripts.run_bus_height_pair import save,sha
setup_env();manifest=json.loads((SOURCE/'manifest.json').read_text());assert manifest['binding']['ready']
selected=pickle.loads((SOURCE/'selected_window.pkl').read_bytes());original=[r for group in selected['groups'] for r in group]
records=pickle.loads((Path(manifest['dataset'])/'labels/data.pkl').read_bytes());assert len(records)==len(original)==144
for raw,new in zip(original,records):
    # Labels are cloned before only data-index and multiview-index remapping.
    a=copy.deepcopy(raw);b=copy.deepcopy(new)
    for key in ('data_index','multiview_start_idx'):a.pop(key,None);b.pop(key,None)
    assert pickle.dumps(a,protocol=4)==pickle.dumps(b,protocol=4),'Source annotations changed'
for part in ('images','hdmaps'):
    env=lmdb.open(str(Path(manifest['dataset'])/part),readonly=True,lock=False,readahead=False)
    with env.begin() as tx:
        for i,expected in enumerate(manifest['image_and_map_hashes'][part]):
            assert hashlib.sha256(tx.get(str(i).encode())).hexdigest()==expected
    env.close()
print('SOURCE_BYTES_PASS: all144 records, images and maps preserved except dataset indices.',flush=True)
b=make_backend(manifest,SOURCE/'input_audit',audit_only=True);g=b.generate(REQUEST,iteration=0)
assert_native(g,manifest);assert 'video' not in g.artifacts and g.metadata['dd2_audit_only']
save(SOURCE/'audit_generation.json',asdict(g))
files=[ROOT/'scripts/run_native_bus_control.py',ROOT/'scripts/preflight_native_bus_control.py',ROOT/'scripts/prepare_native_bus.py',
    SOURCE/'manifest.json',SOURCE/'identity.json',SOURCE/'selection_protocol.json',SOURCE/'selection_ledger.json']
files.extend((ROOT/'driveloop').rglob('*.py'))
for folder in ('dreamer-train/projects/DriveDreamer2','dreamer-datasets/dreamer_datasets'):
    files.extend((ROOT/folder).rglob('*.py'))
files.extend(Path(manifest['dataset'])/part/name for part,name in [('labels','data.pkl'),('images','data.mdb'),('hdmaps','data.mdb')])
save(SOURCE/'preflight.json',{'passed':True,'generation_run':False,'records_verified':144,'runtime_input_audit':g.metadata['dd2_runtime_input_audit'],
    'fixed_hashes':{str(p):sha(p) for p in sorted(set(files))},
    'claim':'Actual targeted loader and cached prompt path audited. Full diffusion remains for user-run command.'})
print('NATIVE_RUNTIME_PREFLIGHT_PASS: no generated video.',flush=True)
