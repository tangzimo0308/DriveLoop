"""CPU-only audit of archived planned geometry against source map/annotations."""
import copy,hashlib,json,pickle
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon,MultiPoint,Point,LineString
from scipy.spatial import ConvexHull
from nuscenes.map_expansion.map_api import NuScenesMap
from dreamer_datasets import load_dataset,boxes3d_utils
from driveloop.dd2_override import apply_dd2_override_to_sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();run=root/'outputs/bus_image_condition_pair_20261006_123610_72403'
out=root/'outputs/revision_20261006_lane_geometry';out.mkdir(exist_ok=True)
summary=json.loads((run/'summary.json').read_text());meta=next(c for c in summary['cases'] if c['actor_injection'])['generation']['metadata']
override=meta['dd2_override_json'];entries=override['boxes3d']['per_frame_append_ego']
labels=Path(meta['dd2_source_sample_binding']['labels_path']);records=pickle.loads(labels.read_bytes());dataset=load_dataset(str(labels.parent.parent))
scene=next(r for r in json.loads(Path('/data/nuscenes/v1.0-trainval/scene.json').read_text()) if r['token']==meta['dd2_source_sample_binding']['selector']['scene_token'])
log=next(r for r in json.loads(Path('/data/nuscenes/v1.0-trainval/log.json').read_text()) if r['token']==scene['log_token'])
m=NuScenesMap(dataroot='/data/nuscenes',map_name=log['location'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corners(box):return boxes3d_utils.boxes3d_to_corners3d(np.asarray(box,dtype=np.float32).reshape(1,9),rot_axis=1)[0]
def global_corners(box,calib):
 c=corners(box);T=np.asarray(calib['ego2global'])@np.asarray(calib['cam2ego']);return (np.c_[c,np.ones(len(c))]@T.T)[:,:3]
def intersects_3d(a,b,tol=1e-5):
 # Convex-polyhedron separating-axis test, including face normals and edge crosses.
 ha,hb=ConvexHull(a),ConvexHull(b)
 def edges(h,p):
  ids={tuple(sorted((int(t[i]),int(t[j])))) for t in h.simplices for i,j in ((0,1),(1,2),(0,2))}
  return [p[j]-p[i] for i,j in ids]
 axes=list(ha.equations[:,:3])+list(hb.equations[:,:3])
 axes.extend(np.cross(x,y) for x in edges(ha,a) for y in edges(hb,b))
 for axis in axes:
  norm=np.linalg.norm(axis)
  if norm<1e-9:continue
  axis=axis/norm;aa=a@axis;bb=b@axis
  if min(aa.max(),bb.max())-max(aa.min(),bb.min())<=tol:return False
 return True
cube=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
assert intersects_3d(cube,cube+[.5,0,0])
assert not intersects_3d(cube,cube+[2.,0,0])
assert not intersects_3d(cube,cube+[0,0,3.])
def footprint(c):return MultiPoint(c[:,:2]).convex_hull
def proj(box,calib):return boxes3d_utils.corners3d_to_corners2d(corners(box)[None],cam_intrinsic=calib['cam_intrinsic'])[0]
def poly2d(box,calib):return MultiPoint(proj(box,calib)).convex_hull
last=entries[-1];ego_xy=np.asarray(last['ref_ego2global'])[:2,3];ego_lane=m.record_on_point(*ego_xy,'lane');assert ego_lane
lane_poly=m.extract_polygon(m.get('lane',ego_lane)['polygon_token'])
# The lane actually occupied by ego in the last five frames; do not infer lane membership from nearest center alone.
source_lanes={}
for e in entries:
 c=(np.asarray(e['ref_ego2global'])@np.r_[e['ego']['center_ego'],1])[:2];token=m.record_on_point(*c,'lane')
 if token:source_lanes[token]=m.extract_polygon(m.get('lane',token)['polygon_token'])
road_lanes={ego_lane:lane_poly,**source_lanes}
rows=[];sheet=Image.new('RGB',(448*4,290*2),'white');d=ImageDraw.Draw(sheet)
fig,axes=plt.subplots(2,4,figsize=(16,9),constrained_layout=True)
centerline=LineString(np.asarray(m.discretize_lanes([ego_lane],.25)[ego_lane])[:,:2])
for step,e in enumerate(entries):
 r=records[e['source_record_indices']['cam_front']];raw=dataset[e['source_record_indices']['cam_front']]
 assert r['frame_idx']==e['frame_idx'] and r['cam_type']=='cam_front'
 updated,audit=apply_dd2_override_to_sample(copy.deepcopy(raw),override);assert len(updated['boxes3d'])==len(r['boxes3d'])+1
 box=updated['boxes3d'][-1];glob=global_corners(box,r['calib']);target=footprint(glob);target_proj=poly2d(box,r['calib']);gc=np.asarray(r['calib']['ego2global'])@np.asarray(r['calib']['cam2ego'])@np.r_[box[:3],1]
 ex=np.asarray(r['calib']['ego2global'])[:2,3];tok=m.record_on_point(*gc[:2],'lane');projected=proj(box,r['calib'])
 row={'step':step,'frame_idx':r['frame_idx'],'timestamp_us':r['cam_timestamp'],'target_center_ego':e['ego']['center_ego'],'target_center_global':gc[:3].tolist(),'target_heading_ego':e['ego']['heading_ego'],'ego_center_lane':m.record_on_point(*ex,'lane'),'target_center_lane':tok,'target_center_in_ego_lane':lane_poly.covers(Point(gc[:2])),'center_distance_to_ego_lane_m':lane_poly.distance(Point(gc[:2])),'center_distance_to_ego_centerline_m':centerline.distance(Point(gc[:2])),'footprint_area_m2':target.area,'footprint_in_ego_lane_fraction':target.intersection(lane_poly).area/target.area,'footprint_global':list(target.exterior.coords),'front_projected_bounds':[float(projected[:,0].min()),float(projected[:,1].min()),float(projected[:,0].max()),float(projected[:,1].max())],'source_vehicle_checks':[]}
 ax=axes.flat[step]
 for t,p in road_lanes.items():
  xy=np.array(p.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='lightblue' if t==ego_lane else 'lightgray',alpha=.55);ax.plot(xy[:,0],xy[:,1],color='gray',lw=.6)
 xy=np.asarray(target.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='red',alpha=.4);ax.plot(*gc[:2],'ro',ms=3);ax.plot(*ex,'k^',ms=5)
 img=raw['image'].convert('RGB').resize((448,252));idraw=ImageDraw.Draw(img)
 for j,(b,lab,inst) in enumerate(zip(r['boxes3d'],r['ori_labels3d'],r['instance_tokens'])):
  if not lab.startswith('vehicle.'):continue
  oc=global_corners(b,r['calib']);op=footprint(oc);ip=poly2d(b,r['calib']);inter=target.intersection(op).area;zover=max(0.,min(glob[:,2].max(),oc[:,2].max())-max(glob[:,2].min(),oc[:,2].min()))
  overlap=target_proj.intersection(ip).area
  row['source_vehicle_checks'].append({'source_box_index':j,'label':lab,'instance_token':inst,'footprint_area_overlap_m2':inter,'vertical_overlap_m':float(zover),'convex_box_intersects_3d':intersects_3d(glob,oc),'footprint_clearance_m':target.distance(op),'projection_iou':overlap/target_proj.union(ip).area,'target_projection_overlap_fraction':overlap/target_proj.area,'source_projection_overlap_fraction':overlap/ip.area,'source_footprint_global':list(op.exterior.coords)})
  xy=np.asarray(op.exterior.coords);ax.plot(xy[:,0],xy[:,1],color='orange',lw=1);ax.text(op.centroid.x,op.centroid.y,str(j),fontsize=7,clip_on=True)
  pts=proj(b,r['calib'])*[448/1600,252/900];hull=MultiPoint(pts).convex_hull
  if hull.geom_type=='Polygon':idraw.line(list(hull.exterior.coords),fill='orange',width=1)
  idraw.text((float(pts[:,0].min()),float(pts[:,1].min())),str(j),fill='orange')
 pts=projected*[448/1600,252/900];idraw.line(list(MultiPoint(pts).convex_hull.exterior.coords),fill='red',width=2)
 x=(step%4)*448;y=(step//4)*290;sheet.paste(img,(x,y+30));d.text((x+3,y+4),f'frame {step} | RED planned bus / ORANGE source vehicles',fill='black')
 ax.set_xlim(gc[0]-12,gc[0]+12);ax.set_ylim(gc[1]-24,gc[1]+12);ax.set_aspect('equal');ax.set_title(f"frame {step}: bus in ego lane {row['footprint_in_ego_lane_fraction']:.1%}");ax.ticklabel_format(useOffset=False);ax.tick_params(labelsize=7)
 rows.append(row)
fig.suptitle('Source-map geometry: RED planned bus; ORANGE source vehicles; BLUE ego lane; triangle ego')
fig.savefig(out/'plan_map_overlap.png',dpi=130);plt.close(fig);sheet.save(out/'planned_source_projection.jpg',quality=95)
report={'audit':'archived_bus_plan_geometry_v1','generation_run':str(run),'cpu_only':True,'scene':scene,'map_location':log['location'],'ego_lane_token':ego_lane,'target_lane_tokens':list(source_lanes),'lane_connections':{t:{'incoming':m.get_incoming_lane_ids(t),'outgoing':m.get_outgoing_lane_ids(t)} for t in road_lanes},'hashes':{'labels':sha(labels),'map':sha('/data/nuscenes/maps/expansion/'+log['location']+'.json'),'summary':sha(run/'summary.json'),'audit_script':sha(__file__)},'limitations':['Source map and annotation geometry, not measured generated-video geometry.','First three ego positions are outside lane polygons (connector area); ego lane is directly observed in frames 3-7.','Source-vehicle collision checks use CAM_FRONT annotations only; absence of overlap does not prove collision-free.','2D projection overlap alone is not a 3D collision.','Convex projected 3D corner footprints include small camera/ego pitch effects.'],'frames':rows}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:
 print(r['step'],'ego fraction',round(r['footprint_in_ego_lane_fraction'],4),'center lane dist',round(r['center_distance_to_ego_lane_m'],3),'overlaps',[(v['source_box_index'],v['label'],round(v['footprint_area_overlap_m2'],3),round(v['projection_iou'],3)) for v in r['source_vehicle_checks'] if v['footprint_area_overlap_m2']>.01 or v['projection_iou']>.05])
print('AUDIT_COMPLETE',out)
