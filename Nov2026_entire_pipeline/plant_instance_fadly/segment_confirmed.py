"""Assign plant instances from axes confirmed in stem_axes_to_review.csv."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
os.environ.setdefault('OMP_NUM_THREADS','2')
HERE=Path(__file__).resolve().parent
from cloud_io import write_ply,instance_colors
from gap_segmentation import Settings,segment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    args=p.parse_args()
    review=HERE/'review'/args.dataset
    manifest=json.loads((review/'review_manifest.json').read_text())
    source_hash=sha256(HERE/'gap_segmentation.py')
    if source_hash!=manifest['code_sha256']:
        raise SystemExit('The detector code changed since review preparation. Run prepare_review.py again and review the new CSV.')
    axes_to_plants={};bases={}
    with (review/'stem_axes_to_review.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try: axis_id=int(row['axis_id'])
            except (ValueError,TypeError): continue
            raw=row.get('plant_id','').strip()
            if not raw: continue
            plant_id=int(raw)
            if plant_id<=0: raise SystemExit('Confirmed plant_id values must be positive integers; leave rejected axes blank.')
            axes_to_plants[axis_id]=plant_id
            bases[axis_id]=np.array([float(row['base_x_m']),float(row['base_y_m'])])
    if not axes_to_plants: raise SystemExit('No plant IDs were entered. Edit stem_axes_to_review.csv first.')
    cache=HERE/'inputs'/args.dataset
    xyz=np.load(cache/'xyz.npy');semantic=np.load(cache/'semantic.npy')
    detail=json.loads((cache/'stem_location_hints.json').read_text())
    result=segment(xyz,semantic,detail['centers_xy'],Settings(),augment=True,bridges=True,
                   weighted=True,axis_groups=axes_to_plants)
    # Safety check: make sure the reviewed base locations still refer to the same candidates.
    current={int(a['axis_id']):np.asarray(a['base_xy']) for a in result['candidate_axes']}
    for axis_id,base in bases.items():
        if axis_id not in current or np.linalg.norm(base-current[axis_id])>.005:
            raise SystemExit(f'Candidate axis {axis_id} changed. Re-run prepare_review.py and check the CSV again.')
    plant_ids=sorted(set(axes_to_plants.values()))
    remap=np.zeros(len(plant_ids)+1,np.int32)
    remap[1:]=plant_ids
    candidate=remap[result['candidate'][result['inverse']]]
    conservative=remap[result['accepted'][result['inverse']]]
    status=result['status'][result['inverse']]
    input_index=np.load(cache/'input_index.npy')
    out=HERE/'results'/args.dataset;out.mkdir(parents=True,exist_ok=True)
    fields={'plant_id':conservative,'candidate_plant_id':candidate,'semantic':semantic,
            'assignment_status':status,'ownership_margin':result['margin'][result['inverse']],
            'input_index':input_index}
    write_ply(out/'plants_conservative.ply',xyz,instance_colors(conservative),fields)
    write_ply(out/'plants_candidate_including_gaps.ply',xyz,instance_colors(candidate),fields)
    colours=np.array([[100,100,100],[38,128,64],[242,140,25],[205,30,35]],np.uint8)
    status_colours=colours[status]
    write_ply(out/'assignment_evidence.ply',xyz,status_colours,fields)
    points=result['points'];ids=np.unique(np.linspace(0,len(points)-1,min(150000,len(points)),dtype=int))
    plot_labels=remap[result['accepted'][ids]]
    fig,ax=plt.subplots(figsize=(16,7))
    ax.scatter(points[ids,1],points[ids,2],c=instance_colors(plot_labels)/255.,s=.65,linewidths=0)
    ax.set_title('Confirmed plant IDs; grey is unassigned or uncertain')
    ax.set_xlabel('Along-row Y (m)');ax.set_ylabel('Height Z (m)');ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(out/'plants_overview.png',dpi=150);plt.close(fig)
    summary={'dataset':args.dataset,'reviewed_axes':axes_to_plants,'plant_ids':plant_ids,
       'work_points':len(xyz),'assigned_points':int((conservative>0).sum()),
       'unassigned_points':int((conservative==0).sum()),
       'status_counts':{str(i):int((status==i).sum()) for i in range(4)},
       'note':'Candidate output may include proposed gap paths. Conservative output only assigns points with an observed local path and a non-ambiguous best plant source.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
