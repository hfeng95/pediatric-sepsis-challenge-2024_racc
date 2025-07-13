#!/usr/bin/env python3

import json, time, argparse, numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, roc_curve
)
from helper_code import load_challenge_data, load_challenge_predictions, read_or_compute_threshold

def calculate_net_benefit(y, p, t):
    tp = np.sum((y==1)&(p>=t)); fp = np.sum((y==0)&(p>=t)); n=len(y)
    return (tp/n) - ((t/(1-t))*(fp/n))

def calculate_ece(probs, labels, n_bins=10):
    edges = np.linspace(0,1,n_bins+1); ece=0.0
    for i in range(n_bins):
        m=(probs>edges[i])&(probs<=edges[i+1])
        if m.sum()>0:
            ece+= m.sum()*abs(labels[m].mean()-probs[m].mean())/len(probs)
    return ece

def read_inference_and_parsimony(fn):
    inf,par=None,None
    for L in open(fn):
        if L.startswith("Average time per patient:"):
            inf=float(L.split(":")[1].split()[0])
        if L.startswith("Parsimony Score:"):
            par=float(L.split(":")[1].strip())
    if inf is None or par is None:
        raise ValueError(f"Missing inference or parsimony in {fn}")
    return inf,par

def load_json(path):
    return json.load(open(path))

def evaluate_model(
    label_folder, output_folder, inference_time_file, threshold_file,
    scale_params_file, factor_loadings_file, zscore_params_file
):
    # 1. load data & predictions
    _,_,y_true,_ = load_challenge_data(label_folder)
    ids,y_prob,_= load_challenge_predictions(output_folder)

    # 2. threshold (must come from threshold.txt, default to 0.5 if missing/unreadable)
    try:
        thr = float(open(threshold_file).read().strip()) if threshold_file else 0.5
    except Exception as e:
        print(f"Warning: could not read threshold_file '{threshold_file}', using default 0.5. Error: {e}")
        thr = 0.5
    y_pred = (y_prob >= thr).astype(int)

    # 3. compute metrics
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
    sens=tp/(tp+fn) if (tp+fn)>0 else np.nan
    spec=tn/(tn+fp) if (tn+fp)>0 else np.nan
    F1=(2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    auc_s=roc_auc_score(y_true,y_prob)
    prec,rec,_=precision_recall_curve(y_true,y_prob)
    auprc=auc(rec,prec)
    nb=calculate_net_benefit(y_true,y_prob,thr)
    ece=calculate_ece(y_prob,y_true)

    # 4. inference & parsimony
    inf_time,par= read_inference_and_parsimony(inference_time_file)

    # 5. base output
    out = {
      "AUC":auc_s,"AUPRC":auprc,"Net Benefit":nb,"ECE":ece,
      "F1":F1,"Sensitivity":sens,"Specificity":spec,
      "Parsimony Score":par,"Inference Time":inf_time,
      "threshold_used":thr,"tp":tp,"fp":fp,"fn":fn,"tn":tn
    }

    # 6. skip weighting if sens<0.8
    if sens < 0.8:
        out["weighted_score"]=None
        out["scaled_weighted_score"]=None
        return out

    # 7. load R-saved params
    sp    = load_json(scale_params_file)     # {"center": [...], "scale":[...]}
    loads = load_json(factor_loadings_file)  # e.g. {"F1":0.648,...}
    zp    = load_json(zscore_params_file)    # {"center":mu_z,"scale":sd_z} or [mu_z,sd_z]

    # 8. standardize factor metrics
    centers = sp["center"]; scales=sp["scale"]
    if not isinstance(centers, dict):
        feats=list(loads.keys())
        centers=dict(zip(feats,centers)); scales=dict(zip(feats,scales))
    raw_factor=0
    for m,loading in loads.items():
        x={"F1":F1,"AUPRC":auprc,"Net.Benefit":nb,"ECE":ece}[m]
        raw_factor+= loading*((x-centers[m])/scales[m])

    # 9. combine parsimony & inference-speed
    wp=0.05; wi=0.05
    inv_par=1-par
    if "Inference Time" not in centers:
        centers["Inference Time"]=sp["center"][-1]
        scales ["Inference Time"]=sp["scale"][-1]
    i_norm=(inf_time-centers["Inference Time"])/scales["Inference Time"]
    i_speed=-i_norm
    raw_combo= raw_factor*(1-wp-wi) + inv_par*wp + i_speed*wi
    raw_combo=round(raw_combo,4)

    #10. final z-score
    if isinstance(zp,dict):
        mu_z=zp["center"] if isinstance(zp["center"],(int,float)) else list(zp["center"].values())[0]
        sd_z=zp["scale"] if isinstance(zp["scale"],(int,float)) else list(zp["scale"].values())[0]
    else:
        mu_z,sd_z=zp
    scaled_w=round((raw_combo-mu_z)/sd_z,4)

    out["weighted_score"]=raw_combo
    out["scaled_weighted_score"]=scaled_w
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("label_folder");p.add_argument("output_folder")
    p.add_argument("inference_time_file")
    p.add_argument("threshold_file", type=str, help="Threshold file for classification (required)")
    p.add_argument("scale_params_json")
    p.add_argument("factor_loadings_json")
    p.add_argument("zscore_params_json")
    p.add_argument("output_file",nargs="?",default=None)
    args=p.parse_args()
    res=evaluate_model(
      args.label_folder,args.output_folder,args.inference_time_file,
      args.threshold_file,args.scale_params_json,
      args.factor_loadings_json,args.zscore_params_json
    )
    # numpy→native
    for k,v in res.items():
        if hasattr(v,"item"): res[k]=v.item()
    payload={"score":res,"completion_time":time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    j=json.dumps(payload)
    if args.output_file: open(args.output_file,"w").write(j)
    else: print(j)
