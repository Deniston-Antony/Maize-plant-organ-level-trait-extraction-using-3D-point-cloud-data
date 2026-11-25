"""Detect candidate maize stem axes and make a human review file.

This is the first stage of the stem-anchored instance workflow. It does not
pretend every detected axis is a biological plant. The reviewer puts the same
positive plant_id on axes that belong to one plant and leaves rejected axes
blank in the CSV.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE=Path(__file__).resolve().parent
# All algorithm code and prepared input data live in this folder.
from cloud_io import write_ply,instance_colors
from gap_segmentation import Settings,segment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    args=parser.parse_args()
    cache=HERE/'inputs'/args.dataset
    xyz=np.load(cache/'xyz.npy')
    semantic=np.load(cache/'semantic.npy')
    detail=json.loads((cache/'stem_location_hints.json').read_text())
    out=HERE/'review'/args.dataset
    if (out/'stem_axes_to_review.csv').exists():
        raise SystemExit(f'Output already exists: {out}. Preserve or rename it before rerunning; the review CSV may contain manual edits.')
    cfg=Settings()
    print(f'Finding candidate stem axes in {args.dataset} ({len(xyz):,} work points)...',flush=True)
    result=segment(xyz,semantic,detail['centers_xy'],cfg,augment=True,bridges=True,weighted=True)
    out=HERE/'review'/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    candidate=result['candidate'][result['inverse']]
    status=result['status'][result['inverse']]
    base_id=np.load(cache/'previous_plant_id.npy')
    input_index=np.load(cache/'input_index.npy')
    fields={'candidate_axis_id':candidate,'previous_plant_id':base_id,
            'semantic':semantic,'assignment_status':status,'input_index':input_index}
    write_ply(out/'candidate_axes_all_points.ply',xyz,instance_colors(candidate),fields)
    stem_mask=semantic==0
    write_ply(out/'candidate_stem_points.ply',xyz[stem_mask],instance_colors(candidate[stem_mask]),
              {k:v[stem_mask] for k,v in fields.items()})
    headers=['axis_id','source','previous_plant_id','base_x_m','base_y_m',
             'vertical_slices','support_nodes','plant_id','review_notes']
    with (out/'stem_axes_to_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=headers);writer.writeheader()
        for axis in result['axes']:
            observed=np.asarray(axis['observed_track']).reshape(-1,3)
            writer.writerow({'axis_id':axis['axis_id'],'source':axis['source'],
                'previous_plant_id':axis['baseline_plant_id'] or '',
                'base_x_m':f"{axis['base_xy'][0]:.4f}",'base_y_m':f"{axis['base_xy'][1]:.4f}",
                'vertical_slices':len(observed),'support_nodes':axis['seed_nodes'],
                'plant_id':'','review_notes':''})
    # Three simple projections show candidates without requiring point-cloud software.
    points=result['points']; axis_labels=result['candidate']
    ids=np.unique(np.linspace(0,len(points)-1,min(120000,len(points)),dtype=int))
    fig,axes=plt.subplots(3,1,figsize=(16,13))
    colours=instance_colors(axis_labels[ids])/255.
    axes[0].scatter(points[ids,1],points[ids,0],c=colours,s=.65,linewidths=0)
    axes[0].set_title('Top view: whole-row candidate groups (each colour is a stem-axis proposal)')
    axes[0].set_ylabel('X (m)')
    axes[1].scatter(points[ids,1],points[ids,2],c=colours,s=.65,linewidths=0)
    axes[1].set_title('Side view: whole row')
    axes[1].set_ylabel('Height Z (m)')
    stem=result['semantic']==0
    axes[2].scatter(points[stem,1],points[stem,0],c=instance_colors(axis_labels[stem])/255.,s=.8,linewidths=0)
    for axis in result['axes']:
        x,y=axis['base_xy']
        axes[2].annotate(str(axis['axis_id']),(y,x),fontsize=8,
                         xytext=(3,3),textcoords='offset points',color='black',
                         bbox={'boxstyle':'round,pad=.15','fc':'white','alpha':.7,'ec':'none'})
    axes[2].set_title('Top view: stem-class points; numbers are candidate axis IDs from the CSV')
    axes[2].set_ylabel('X (m)')
    for ax in axes:
        ax.set_aspect('equal')
    axes[-1].set_xlabel('Along-row Y (m)')
    fig.tight_layout();fig.savefig(out/'stem_axis_review.png',dpi=150);plt.close(fig)
    # A simpler base-only view makes close, competing hypotheses easier to
    # point out than the full-cloud colour preview.
    fig,ax=plt.subplots(figsize=(15,5))
    bg=points[ids]
    ax.scatter(bg[:,1],bg[:,0],c='#d7d7d7',s=.55,linewidths=0,zorder=1)
    row=result['source_diagnostics'].get('row_fit_x_from_y')
    if row is not None:
        row_y=np.linspace(float(points[:,1].min()),float(points[:,1].max()),200)
        ax.plot(row_y,np.polyval(row,row_y),color='#2878b5',linestyle='--',
                linewidth=1.5,label='estimated crop row',zorder=2)
    for i,axis in enumerate(result['axes']):
        x,y=axis['base_xy']
        is_extra=axis['source']=='additional_stem_axis_hypothesis'
        colour='#e67e22' if is_extra else '#198754'
        ax.scatter([y],[x],s=65,c=colour,edgecolors='black',linewidths=.6,zorder=4)
        dy=.035 if i%2==0 else -.035
        ax.annotate(str(axis['axis_id']),(y+dy,x+(.025 if i%2==0 else -.025)),
                    fontsize=9,fontweight='bold',color='black',ha='center',
                    bbox={'boxstyle':'round,pad=.18','fc':'white','alpha':.9,'ec':'none'},zorder=5)
    ax.scatter([],[],s=55,c='#198754',edgecolors='black',label='old anchor with upright-track support')
    ax.scatter([],[],s=55,c='#e67e22',edgecolors='black',label='new stem candidate')
    ax.set_title('Candidate stem bases only — IDs match the review CSV')
    ax.set_xlabel('Along-row Y (m)');ax.set_ylabel('Cross-row X (m)')
    ax.set_aspect('equal');ax.legend(loc='upper left',ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(out/'candidate_bases.png',dpi=170);plt.close(fig)
    crops=out/'axis_crops';crops.mkdir(exist_ok=True)
    for axis in result['axes']:
        x,y=axis['base_xy']
        roi=(np.abs(points[:,0]-x)<=.18)&(np.abs(points[:,1]-y)<=.18)
        local=np.flatnonzero(roi)
        if not len(local): continue
        take=local[np.unique(np.linspace(0,len(local)-1,min(30000,len(local)),dtype=int))]
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        c=instance_colors(axis_labels[take])/255.
        axs[0].scatter(points[take,1],points[take,0],c=c,s=2,linewidths=0)
        axs[0].scatter([y],[x],marker='x',c='black',s=60)
        axs[0].set_title(f"Top view around candidate axis {axis['axis_id']}")
        axs[0].set_xlabel('Y (m)');axs[0].set_ylabel('X (m)')
        axs[1].scatter(points[take,1],points[take,2],c=c,s=2,linewidths=0)
        axs[1].set_title('Side view of the same local area')
        axs[1].set_xlabel('Y (m)');axs[1].set_ylabel('Height Z (m)')
        for ax in axs: ax.set_aspect('equal')
        fig.tight_layout();fig.savefig(crops/f"axis_{int(axis['axis_id']):02d}.png",dpi=140);plt.close(fig)
    manifest={'dataset':args.dataset,'work_points':len(xyz),'candidate_axes':len(result['axes']),
        'input_work_config':json.loads((cache/'work_config.json').read_text()),
        'semantic_config':json.loads((cache/'semantic_config.json').read_text()),
        'settings':result['settings'],'source_diagnostics':result['source_diagnostics'],
        'code_sha256':sha256(HERE/'gap_segmentation.py'),
        'instruction':'Review candidate axes. Fill plant_id with a positive integer for every confirmed axis. Use the same plant_id for axes believed to belong to one plant. Leave rejected or unresolved axes blank.'}
    (out/'review_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"Found {len(result['axes'])} candidate axes. Review: {out/'stem_axes_to_review.csv'}",flush=True)

if __name__=='__main__': main()

