"""Conservative scene exclusions from recorded generation evidence and named experiments."""
import json,re,hashlib
from pathlib import Path
ROOT=Path('/home/lesliton/projects/DriveLoop');OUT=ROOT/'outputs/revision_20261015_history_inventory';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=ROOT/'outputs/revision_20261006_source_screen_expanded/shortlist.json';mapping={r['candidate_index']:r['scene_token'] for r in json.loads(old.read_text())['ledger']}
exclusions={};unresolved=[];inputs={};count=0;gen_count=0
pattern=re.compile(r'candidate[_ -]?(\d+)',re.I)
def add(scene,why,file):
 if isinstance(scene,str) and re.fullmatch('[0-9a-f]{32}',scene):
  row={'reason':why,'path':str(file)}
  if row not in exclusions.setdefault(scene,[]):exclusions[scene].append(row)
def scenes(x):
 s=set()
 if isinstance(x,dict):
  for k,v in x.items():
   if k=='scene_token' and isinstance(v,str):s.add(v)
   elif k in ['scene_tokens','matched_scene_tokens'] and isinstance(v,list):s.update(t for t in v if isinstance(t,str))
   else:s.update(scenes(v))
 elif isinstance(x,list):
  for v in x:s.update(scenes(v))
 return s

def walk(x,file):
 global gen_count
 if isinstance(x,dict):
  art=x.get('artifacts');meta=x.get('metadata',{})
  if isinstance(art,dict) and isinstance(art.get('video'),str) and isinstance(meta,dict) and not meta.get('dd2_audit_only',False):
   gen_count+=1;ss=scenes(meta)
   for sc in ss:add(sc,'recorded_generation_metadata',file)
   ids=pattern.findall(str(file)+' '+art['video'])
   for n in ids:
    if int(n) in mapping:add(mapping[int(n)],'generation_path_candidate_mapping',file)
   if not ss and not any(int(n) in mapping for n in ids):unresolved.append({'json':str(file),'video':art['video']})
   return
  for v in x.values():walk(v,file)
 elif isinstance(x,list):
  for v in x:walk(v,file)
errors=[]
files=sorted((ROOT/'outputs').rglob('*.json'))
for f in files:
 if f.is_relative_to(OUT):continue
 try:
  data=json.loads(f.read_text());inputs[str(f)]=sha(f);count+=1;walk(data,f)
 except (ValueError,UnicodeError,OSError) as e:errors.append({'path':str(f),'error':str(e)})
for f in sorted((ROOT/'experiments').iterdir()):
 if f.is_file():
  ids=pattern.findall(f.name)
  if ids:inputs[str(f)]=sha(f)
  for n in ids:
   if int(n) in mapping:add(mapping[int(n)],'named_experiment_conservative_exclusion',f)
for sc,why in [('6f5133fe62b240e797bac25aeff8b531','previous_candidate28'),('de943e246dad4ad686de98008a634ecf','previous_candidate91'),('796e7ae080984a0ab2b96ce80f18ca3b','previous_candidate5596'),('5e270cfc71714e2bb94bad445c6159a4','current_development_candidate1583')]:add(sc,why,'known_session_history')
unresolved=list({(x['json'],x['video']):x for x in unresolved}.values())
out={'schema':'recorded_scene_exclusion_v1','json_files_scanned':count,'generation_objects':gen_count,'excluded_scenes':exclusions,'unresolved_generation_records':unresolved,'parse_errors':errors,'input_hashes':inputs,'candidate_scene_mapping_sha256':sha(old),'limits':['Known-recorded-history exclusion only; absent, deleted or overwritten records cannot be reconstructed.','Raw DD2 output directories contain rolling outputs without full history metadata.','Unresolved generation identity prevents claiming complete scene-disjoint held-out status.','Named experiment exclusions are conservative, including source-only experiments.'],'generation_run':False}
(OUT/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
print('HISTORY',count,gen_count,'excluded',len(exclusions),'unresolved',len(unresolved),'parse_errors',errors)
print('EXCLUDED_CANDIDATES',[(n,s) for n,s in mapping.items() if s in exclusions][:120])
print('UNRESOLVED_PREVIEW',unresolved[:8])
