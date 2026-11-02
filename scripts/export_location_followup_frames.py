"""Export all fixed sampled reference PNGs; no generation or detection."""
import json,hashlib,shutil
from pathlib import Path
from scripts.run_native_matched_capture import validate_hooks
ROOT=Path('/home/lesliton/projects/DriveLoop')
P=ROOT/'outputs/revision_20261019_location_followup_sampling.json'
OUT=ROOT/'outputs/revision_20261019_location_followup_frames'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=read(P);rows=[];receipts={}
assert protocol['seed_offsets']==[1,2] and protocol['frame_indices']==[4,5,6,7]
for entry in protocol['runs']:
    run=Path(entry['run']);summary=read(run/'summary.json');assert summary['complete']
    receipts[str(run/'summary.json')]=sha(run/'summary.json')
    for seed in protocol['seed_offsets']:
        receipt=run/'references'/f'iteration_{seed:02d}_original'/'capture_receipt.json'
        c=read(receipt);g=c['generation'];q=g['metadata']['native_phase_contract']
        assert g['iteration']==seed and q['phase_s']==0
        assert sha(g['artifacts']['video'])==c['video_sha256']
        h=validate_hooks(Path(c['capture_dir']));receipts[str(receipt)]=sha(receipt)
        for j in protocol['frame_indices']:
            png=Path(c['capture_dir'])/'generated_front'/f'frame{j}.png';digest=sha(png)
            assert digest==h['files'][f'generated_front/frame{j}.png']
            assert digest not in protocol['known_local_png_hashes'],('Previously available frame',str(png))
            rows.append({'scene_key':entry['scene_key'],'seed':seed,'frame':j,'scene_token':q['scene_token'],'instance_token':q['instance_token'],'registry_sha256':q['registry_sha256'],'source_path':str(png),'sha256':digest,'receipt_path':str(receipt),'receipt_sha256':sha(receipt),'filename':digest+'.png'})
assert len(rows)==16 and len({x['sha256'] for x in rows})==16
OUT.mkdir(exist_ok=False)
for r in rows:
    shutil.copyfile(r['source_path'],OUT/r['filename']);assert sha(OUT/r['filename'])==r['sha256']
manifest={'complete':True,'generation_run':False,'detector_inference_run':False,'protocol_sha256':sha(P),'frames':rows,'receipt_hashes':receipts,'unique_frames':16,'limits':protocol['limits']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EXPORTED_FIXED_FOLLOWUP',len(rows),OUT)
