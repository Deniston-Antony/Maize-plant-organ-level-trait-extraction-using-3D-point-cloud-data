"""Experimental multi-axis, gap-aware plant ownership on OBSERVED points.

Engineering adaptation of stem-seeded graph competition and geometric
continuity ideas, NOT a reproduction of a named paper or a completion network.
No new surface points are generated. Distances/margins are not probabilities.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, block_diag
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.optimize import linear_sum_assignment


@dataclass
class Settings:
    graph_voxel: float = .008
    neighbours: int = 18
    local_radius: float = .022
    max_gap: float = .070
    normal_penalty: float = 3.
    gap_multiplier: float = 3.
    margin_threshold: float = .12
    minimum_cost_difference: float = .015
    slice_height: float = .015
    slice_cluster_radius: float = .014
    track_max_gap: float = .075
    minimum_track_span: float = .12
    minimum_axis_separation: float = .035
    row_max_residual: float = .12
    stem_tangent_z_min: float = .65
    stem_linearity_min: float = .25
    minimum_seed_nodes: int = 18


def voxel_sample(xyz, semantic, voxel):
    origin = xyz.min(axis=0).astype(float)
    cells = np.floor((xyz.astype(float)-origin)/voxel).astype(np.int64)
    dims = cells.max(axis=0)+1
    if int(dims[0])*int(dims[1])*int(dims[2]) >= np.iinfo(np.int64).max:
        raise ValueError('Voxel keys overflow; inspect coordinate units.')
    key = (cells[:,0]*dims[1]+cells[:,1])*dims[2]+cells[:,2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    n = len(first)
    stem = np.bincount(inverse, weights=(semantic==0), minlength=n)
    leaf = np.bincount(inverse, weights=(semantic==1), minlength=n)
    sem = np.full(n,-1,np.int8)
    known = stem+leaf
    sem[(known>0)&(stem>=.65*known)] = 0
    sem[(known>0)&(leaf>=.65*known)] = 1
    return xyz[first].astype(float), sem, first, inverse


def local_geometry(xyz, neighbours=18):
    n=len(xyz)
    k=min(max(neighbours,4),n)
    normal=np.zeros((n,3),np.float32)
    tangent=np.zeros((n,3),np.float32)
    planar=np.zeros(n,np.float32)
    linear=np.zeros(n,np.float32)
    tree=cKDTree(xyz)
    for start in range(0,n,15000):
        stop=min(start+15000,n)
        _, index=tree.query(xyz[start:stop],k=k,workers=2)
        nb=xyz[index]
        centred=nb-nb.mean(axis=1,keepdims=True)
        values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centred,centred)/k)
        values=np.maximum(values,1e-12)
        normal[start:stop]=vectors[:,:,0]
        tangent[start:stop]=vectors[:,:,2]
        planar[start:stop]=(values[:,1]-values[:,0])/values[:,2]
        linear[start:stop]=(values[:,2]-values[:,1])/values[:,2]
    return dict(normal=normal,tangent=tangent,planarity=planar,linearity=linear)


def _groups_xy(points,radius):
    if len(points)<3:
        return []
    pairs=cKDTree(points[:,:2]).query_pairs(radius,output_type='ndarray')
    adjacency=coo_matrix((np.ones(len(pairs),np.int8),(pairs[:,0],pairs[:,1])),
                         shape=(len(points),len(points))).tocsr()
    _, labels=connected_components(adjacency,directed=False)
    return [np.flatnonzero(labels==i) for i,c in enumerate(np.bincount(labels)) if c>=3]


def find_stem_tracks(xyz,semantic,geometry,cfg):
    """Slice-wise tracking; close stems are NOT globally merged at 12 cm.

    A track is a stem-axis HYPOTHESIS. Its biological plant identity is unknown.
    Short gaps are tolerated in linking slices; acceptance requires observed
    support at multiple heights. No fabricated points count as support.
    """
    low=float(np.quantile(xyz[:,2],.02))
    ztop=low+min(.55,.8*float(np.ptp(xyz[:,2])))
    # A semantic "stem" label by itself is not enough: soil leftovers and
    # leaves can also receive that class. Keep points with upright, line-like
    # local shape. The stricter geometry-only route can recover missed stems,
    # but those proposals are checked against the crop row below.
    tangent_z=np.abs(geometry['tangent'][:,2])
    semantic_stem=(semantic==0)&(tangent_z>=cfg.stem_tangent_z_min)&(
        geometry['linearity']>=cfg.stem_linearity_min)
    extra=(geometry['linearity']>.70)&(tangent_z>.90)
    selected=np.flatnonzero((semantic_stem|extra)&(xyz[:,2]<=ztop))
    if not len(selected):
        return [],[]
    bins=np.floor((xyz[selected,2]-low)/cfg.slice_height).astype(int)
    tracks=[]
    for band in np.unique(bins):
        source=selected[bins==band]
        groups=_groups_xy(xyz[source],cfg.slice_cluster_radius)
        detections=[source[g] for g in groups]
        centres=np.array([np.median(xyz[g],axis=0) for g in detections]).reshape(-1,3)
        if not len(centres):
            continue
        z=float(np.median(centres[:,2]))
        active=[i for i,t in enumerate(tracks) if 0<z-t['centres'][-1][2]<=cfg.track_max_gap]
        matched=set()
        if active:
            cost=np.full((len(active),len(centres)),1e6)
            for row,index in enumerate(active):
                history=np.asarray(tracks[index]['centres'][-5:])
                last=history[-1]
                dz=z-last[2]
                slope=np.zeros(2)
                if len(history)>=3 and np.ptp(history[:,2])>.02:
                    slope=np.array([np.polyfit(history[:,2],history[:,a],1)[0] for a in [0,1]])
                    norm=np.linalg.norm(slope)
                    if norm>.7:
                        slope*=.7/norm
                distance=np.linalg.norm(centres[:,:2]-(last[:2]+slope*dz),axis=1)
                gate=.018+.35*dz
                cost[row]=np.where(distance<=gate,distance+.05*dz,1e6)
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if cost[row,col]>=1e5:
                    continue
                t=tracks[active[row]]
                t['centres'].append(centres[col])
                t['nodes'].append(detections[col])
                matched.add(int(col))
        for i in range(len(centres)):
            if i not in matched:
                tracks.append({'centres':[centres[i]],'nodes':[detections[i]]})
    accepted,records=[],[]
    for t in tracks:
        p=np.asarray(t['centres'])
        nodes=np.unique(np.concatenate(t['nodes']))
        span=float(np.ptp(p[:,2]))
        bottom=float(p[:,2].min())
        fit_error=float('inf')
        if len(p)>=3 and span>.02:
            fit=np.column_stack([np.polyval(np.polyfit(p[:,2],p[:,a],1),p[:,2]) for a in [0,1]])
            fit_error=float(np.sqrt(np.mean(np.sum((p[:,:2]-fit)**2,axis=1))))
        good=(len(p)>=6 and len(nodes)>=18 and span>=cfg.minimum_track_span
              and bottom<=low+.16 and fit_error<=.020)
        basal=p[p[:,2]<=bottom+.045]
        base=np.median(basal[:,:2],axis=0)
        record={'base_xy':base.tolist(),'min_z':bottom,'max_z':float(p[:,2].max()),
                'span_m':span,'observed_slices':len(p),'support_nodes':len(nodes),
                'fit_rmse_m':fit_error if np.isfinite(fit_error) else None,
                'accepted_geometric_hypothesis':bool(good),
                'biological_plant_identity_verified':False}
        records.append(record)
        if good:
            accepted.append(dict(base=base,centres=p,nodes=nodes,record=record))
    # Eliminate near-coincident duplicate surface tracks only at a much finer
    # scale than the old 12 cm merge. Kept hypotheses remain available to review.
    accepted.sort(key=lambda t:(-t['record']['span_m'],-len(t['nodes'])))
    kept=[]
    for t in accepted:
        if any(np.linalg.norm(t['base']-q['base'])<.018 for q in kept):
            t['record']['duplicate_surface_track']=True
        else:
            kept.append(t)
    return sorted(kept,key=lambda t:t['base'][1]),records


def make_sources(xyz,semantic,tracks,baseline_centres,cfg,augment=True):
    """Use C as an explicit prior; optionally add independent supported axes."""
    centres=np.asarray(baseline_centres,dtype=float).reshape(-1,2)
    # The scan follows one crop row. Fit its path robustly from the existing
    # centres so an upright leaf away from the row cannot become a new plant.
    row_fit=None
    if len(centres)>=4:
        good=np.ones(len(centres),bool)
        for _ in range(3):
            row_fit=np.polyfit(centres[good,1],centres[good,0],1)
            residual=np.abs(centres[:,0]-np.polyval(row_fit,centres[:,1]))
            limit=max(.06,4*float(np.median(residual[good])))
            updated=residual<=limit
            if updated.sum()<3 or np.array_equal(updated,good):
                break
            good=updated
        row_fit=np.polyfit(centres[good,1],centres[good,0],1)
    rejected_baseline=[]
    rejected_off_row=[]
    axes=[]
    matched={}
    if len(centres) and tracks:
        sep=np.linalg.norm(centres[:,None,:]-np.array([t['base'] for t in tracks])[None,:,:],axis=2)
        rows,cols=linear_sum_assignment(sep)
        matched={int(r):int(c) for r,c in zip(rows,cols) if sep[r,c]<=.09}
    for i,center in enumerate(centres):
        t=tracks[matched[i]] if i in matched else None
        if t is not None:
            axes.append(dict(base=t['base'],nodes=t['nodes'],centres=t['centres'],
                             origin='C_anchor_refined_by_track',baseline_id=i+1))
        else:
            # An old centre without visible upright stem support is only a
            # location hint. Seeding it anyway produced soil/leaf false plants.
            rejected_baseline.append({'previous_plant_id':i+1,'base_xy':center.tolist(),
                                      'reason':'no_supported_upright_stem_track'})
    if augment:
        used=set(matched.values())
        for i,t in enumerate(tracks):
            if i in used:
                continue
            if row_fit is not None:
                row_residual=float(abs(t['base'][0]-np.polyval(row_fit,t['base'][1])))
                t['record']['row_residual_m']=row_residual
                if row_residual>cfg.row_max_residual:
                    t['record']['rejected_off_row']=True
                    rejected_off_row.append({'base_xy':t['base'].tolist(),
                                             'row_residual_m':row_residual})
                    continue
            if axes and min(np.linalg.norm(t['base']-a['base']) for a in axes)<cfg.minimum_axis_separation:
                continue
            axes.append(dict(base=t['base'],nodes=t['nodes'],centres=t['centres'],
                             origin='additional_stem_axis_hypothesis',baseline_id=None))
    if not axes:
        return [],[],{'row_fit_x_from_y':row_fit.tolist() if row_fit is not None else None,
                      'rejected_baseline_centres':rejected_baseline,
                      'rejected_off_row_tracks':rejected_off_row}
    axes.sort(key=lambda a:a['base'][1])
    bases=np.array([a['base'] for a in axes])
    # No node may seed two competing sources. Resolve overlap by base distance.
    seeds=[]
    records=[]
    kept_axes=[]
    rejected_low_support=[]
    for i,a in enumerate(axes):
        nodes=a['nodes']
        nearest=np.argmin(np.linalg.norm(xyz[nodes,None,:2]-bases[None,:,:],axis=2),axis=1)
        nodes=nodes[nearest==i]
        if len(nodes)<cfg.minimum_seed_nodes:
            rejected_low_support.append({'base_xy':a['base'].tolist(),
                                         'seed_nodes_after_overlap':int(len(nodes))})
            continue
        seeds.append(nodes)
        kept_axes.append(a)
        records.append({'axis_id':len(seeds),'base_xy':a['base'].tolist(),
                        'source':a['origin'],'baseline_plant_id':a['baseline_id'],
                        'seed_nodes':len(nodes),'observed_track':a['centres'].tolist(),
                        'plant_identity_verified':False})
    axes=kept_axes
    return seeds,records,{'row_fit_x_from_y':row_fit.tolist() if row_fit is not None else None,
                          'rejected_baseline_centres':rejected_baseline,
                          'rejected_off_row_tracks':rejected_off_row,
                          'rejected_low_seed_support':rejected_low_support}


def _edge_cost(xyz,rows,cols,geometry,penalty):
    delta=xyz[cols]-xyz[rows]
    distance=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(distance[:,None],1e-12)
    normal=geometry['normal']
    reliability=np.minimum(geometry['planarity'][rows],geometry['planarity'][cols])
    across=.5*(np.abs(np.sum(direction*normal[rows],axis=1))+
               np.abs(np.sum(direction*normal[cols],axis=1)))
    turn=1-np.abs(np.sum(normal[rows]*normal[cols],axis=1))
    return distance*(1+penalty*reliability*(across+turn)),distance


def build_local_graph(xyz,geometry,cfg,weighted=True):
    k=min(cfg.neighbours+1,len(xyz))
    tree=cKDTree(xyz)
    allrows,allcols,allcost=[],[],[]
    for start in range(0,len(xyz),20000):
        stop=min(start+20000,len(xyz))
        d,nb=tree.query(xyz[start:stop],k=k,workers=2)
        rows=np.repeat(np.arange(start,stop),k-1)
        cols=nb[:,1:].ravel()
        dist=d[:,1:].ravel()
        keep=(dist>0)&(dist<=cfg.local_radius)
        rows,cols=rows[keep],cols[keep]
        cost,_=_edge_cost(xyz,rows,cols,geometry,cfg.normal_penalty if weighted else 0.)
        allrows.append(rows);allcols.append(cols);allcost.append(cost)
    graph=coo_matrix((np.concatenate(allcost),(np.concatenate(allrows),np.concatenate(allcols))),
                     shape=(len(xyz),len(xyz))).tocsr()
    graph=graph.maximum(graph.T)
    return graph


def find_gap_edges(xyz,local_graph,geometry,cfg):
    """Limited, geometry-gated intercomponent links; never add surface points."""
    nc,component=connected_components(local_graph,directed=False)
    sizes=np.bincount(component)
    samples=[]
    order=np.argsort(component,kind='stable')
    offsets=np.r_[0,np.cumsum(sizes)]
    for c in np.flatnonzero(sizes>=8):
        ids=order[offsets[c]:offsets[c+1]]
        p=xyz[ids]
        # Spatial extrema plus evenly spaced input representatives; extrema
        # expose fragment boundaries without a full quadratic component search.
        extreme=np.r_[np.argmin(p,axis=0),np.argmax(p,axis=0),
                      np.linspace(0,len(ids)-1,min(16,len(ids)),dtype=int)]
        samples.extend(ids[np.unique(extreme)].tolist())
    samples=np.unique(samples)
    candidates={}
    tree=cKDTree(xyz)
    k=min(192,len(xyz))
    for start in range(0,len(samples),2000):
        ids=samples[start:start+2000]
        distance,near=tree.query(xyz[ids],k=k,distance_upper_bound=cfg.max_gap,workers=2)
        for i,node in enumerate(ids):
            valid=(near[i]<len(xyz))&(distance[i]>0)&np.isfinite(distance[i])
            nbr=near[i,valid]
            ds=distance[i,valid]
            keep=(component[nbr]!=component[node])&(sizes[component[nbr]]>=8)
            nbr,ds=nbr[keep],ds[keep]
            if not len(nbr):
                continue
            # At most the closest candidate in each of four different components.
            seen=set()
            for other,d in zip(nbr,ds):
                c=int(component[other])
                if c in seen:
                    continue
                seen.add(c)
                key=tuple(sorted((int(node),int(other))))
                candidates[key]=float(d)
                if len(seen)>=4:
                    break
    accepted=[]
    for (a,b),distance in candidates.items():
        delta=(xyz[b]-xyz[a])/distance
        pa,pb=geometry['planarity'][[a,b]]
        la,lb=geometry['linearity'][[a,b]]
        na,nb=geometry['normal'][[a,b]]
        ta,tb=geometry['tangent'][[a,b]]
        if min(pa,pb)>.25:
            if abs(np.dot(na,nb))<.70:
                continue
            if max(abs(np.dot(delta,na)),abs(np.dot(delta,nb)))>.45:
                continue
        elif min(la,lb)<.45:
            continue  # weak shape evidence: preserve uncertainty instead
        if la>.50 and abs(np.dot(delta,ta))<.60:
            continue
        if lb>.50 and abs(np.dot(delta,tb))<.60:
            continue
        cost,_=_edge_cost(xyz,np.array([a]),np.array([b]),geometry,cfg.normal_penalty)
        accepted.append((a,b,float(cost[0]*cfg.gap_multiplier+.015),distance))
    # Keep a few lowest-cost connections per component pair; do not turn a
    # missing patch into an unconstrained web of spatial shortcuts.
    accepted.sort(key=lambda e:e[2])
    counts={}
    kept=[]
    for edge in accepted:
        a,b,_,_=edge
        pair=tuple(sorted((int(component[a]),int(component[b]))))
        if counts.get(pair,0)<3:
            kept.append(edge)
            counts[pair]=counts.get(pair,0)+1
    return np.asarray(kept,dtype=float).reshape(-1,4),component,sizes


def add_gap_edges(local_graph,edges):
    if not len(edges):
        return local_graph.copy()
    a,b=edges[:,:2].astype(int).T
    bridge=coo_matrix((edges[:,2],(a,b)),shape=local_graph.shape).tocsr()
    bridge=bridge.maximum(bridge.T)
    # Components differ before bridges, so sums cannot duplicate local edges.
    return local_graph+bridge


def compete(graph,seeds,cfg,local_components=None):
    """Two-best source costs, explicit abstention, no nearest-stem fallback."""
    n=graph.shape[0]
    if not seeds:
        return dict(candidate=np.zeros(n,np.int32),accepted=np.zeros(n,np.int32),
                    margin=np.zeros(n,np.float32),status=np.zeros(n,np.int8),
                    best_cost=np.full(n,np.inf))
    expanded=block_diag((graph,coo_matrix((len(seeds),len(seeds)))),format='csr')
    rows=np.concatenate([np.full(len(s),n+i) for i,s in enumerate(seeds)])
    cols=np.concatenate(seeds)
    connector=coo_matrix((np.full(len(cols),1e-6),(rows,cols)),shape=expanded.shape).tocsr()
    # Source hubs point OUTWARD only. Allowing a reverse connector would let
    # one source teleport through a competing stem's hub at near-zero cost.
    expanded=expanded+connector
    best=np.full(n,np.inf)
    second=np.full(n,np.inf)
    label=np.zeros(n,np.int32)
    for i in range(len(seeds)):
        distance=dijkstra(expanded,directed=True,indices=n+i)[:n]
        better=distance<best
        second=np.where(better,best,np.minimum(second,distance))
        best=np.minimum(best,distance)
        label[better]=i+1
    finite=np.isfinite(best)
    margin=np.zeros(n,np.float32)
    both=finite&np.isfinite(second)
    margin[finite&~np.isfinite(second)]=1
    margin[both]=np.maximum(0,(second[both]-best[both])/np.maximum(second[both],1e-9))
    ambiguous=np.zeros(n,bool)
    ambiguous[both]=(margin[both]<cfg.margin_threshold)|((second[both]-best[both])<cfg.minimum_cost_difference)
    status=np.zeros(n,np.int8)
    status[finite]=1
    if local_components is not None:
        seeded=np.unique(np.concatenate([local_components[s] for s in seeds]))
        requires_gap=finite&~np.isin(local_components,seeded)
        status[requires_gap]=2
    status[ambiguous]=3
    # A point whose only route to a stem crosses a virtual gap is a proposal,
    # not an observed connection. Keep it in ``candidate`` for inspection,
    # but leave it unassigned in the conservative result. The same applies to
    # points with two nearly equal source paths.
    accepted=label.copy()
    accepted[status!=1]=0
    label[~finite]=0
    return dict(candidate=label,accepted=accepted,margin=margin,status=status,best_cost=best)


def segment(xyz,semantic,baseline_centres,cfg=None,augment=True,bridges=True,weighted=True,
            axis_groups=None):
    cfg=cfg or Settings()
    points,sem,first,inverse=voxel_sample(xyz,semantic,cfg.graph_voxel)
    geometry=local_geometry(points,cfg.neighbours)
    tracks,track_records=find_stem_tracks(points,sem,geometry,cfg)
    seeds,axes,source_diagnostics=make_sources(points,sem,tracks,baseline_centres,cfg,augment=augment)
    detected_axis_count=len(axes)
    detected_axes=axes
    if axis_groups is not None:
        # The automatic detector proposes stem axes. A reviewer supplies
        # axis_id -> plant_id; axes assigned the same plant_id share ownership.
        by_plant={}
        axis_by_id={int(a['axis_id']):i for i,a in enumerate(axes)}
        for axis_id,plant_id in axis_groups.items():
            axis_id,plant_id=int(axis_id),int(plant_id)
            if plant_id<=0 or axis_id not in axis_by_id:
                continue
            by_plant.setdefault(plant_id,[]).append(axis_by_id[axis_id])
        grouped_seeds=[]
        grouped_axes=[]
        for plant_id,member_ids in sorted(by_plant.items()):
            grouped_seeds.append(np.unique(np.concatenate([seeds[i] for i in member_ids])))
            grouped_axes.append({'plant_id':plant_id,
                'member_axis_ids':[int(axes[i]['axis_id']) for i in member_ids],
                'base_xy':np.mean([axes[i]['base_xy'] for i in member_ids],axis=0).tolist(),
                'seed_nodes':int(len(grouped_seeds[-1]))})
        seeds,axes=grouped_seeds,grouped_axes
    local=build_local_graph(points,geometry,cfg,weighted=weighted)
    if bridges:
        gap_edges,components,sizes=find_gap_edges(points,local,geometry,cfg)
        graph=add_gap_edges(local,gap_edges)
    else:
        gap_edges=np.empty((0,4))
        _,components=connected_components(local,directed=False)
        sizes=np.bincount(components)
        graph=local
    result=compete(graph,seeds,cfg,local_components=components)
    result.update(points=points,semantic=sem,first=first,inverse=inverse,
                  tracks=track_records,axes=axes,candidate_axes=detected_axes,gap_edges=gap_edges,
                  source_diagnostics=source_diagnostics,
                  settings=asdict(cfg),local_components=components,geometry=geometry,
                  diagnostics={'graph_nodes':len(points),'local_components':len(sizes),
                    'local_components_8plus':int((sizes>=8).sum()),'graph_directed_edges':graph.nnz,
                    'accepted_virtual_edges':len(gap_edges),'stem_axis_hypotheses':detected_axis_count,
                    'confirmed_plant_groups':len(axes) if axis_groups is not None else None,
                    'new_axis_hypotheses':sum(a['source']=='additional_stem_axis_hypothesis' for a in detected_axes),
                    'candidate_coverage':float((result['candidate']>0).mean()),
                    'conservative_coverage':float((result['accepted']>0).mean()),
                    'gap_required_fraction':float((result['status']==2).mean()),
                    'ambiguous_fraction':float((result['status']==3).mean()),
                    'interpretation':'Axes and gap links are hypotheses, not verified biological plants or reconstructed surfaces.'})
    return result
