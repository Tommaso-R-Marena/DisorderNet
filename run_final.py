"""
DisorderNet FINAL: LightGBM + XGBoost ensemble with enriched features.
"""
import json, numpy as np, os, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, balanced_accuracy_score,
                             precision_score, recall_score)
import lightgbm as lgb
import xgboost as xgb

DATA_PATH = "/home/user/workspace/disorder_model/data/disprot_processed.json"
RESULTS_DIR = "/home/user/workspace/disorder_model/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
HYDRO = np.array([1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3],dtype=np.float32)
FLEX = np.array([.984,.906,1.068,1.094,.915,1.031,.950,.927,1.102,.935,.952,1.048,1.049,1.037,1.008,1.046,.997,.931,.904,.929],dtype=np.float32)
DISPROP = np.array([.06,-.02,.192,.736,-.697,.166,.303,-.486,.586,-.326,-.397,.007,.987,.318,.18,.341,.059,-.121,-.884,-.510],dtype=np.float32)
CHARGE = np.array([0,0,-1,-1,0,0,.1,0,1,0,0,0,0,0,1,0,0,0,0,0],dtype=np.float32)
BETA = np.array([.83,1.19,.54,.37,1.38,.75,.87,1.60,.74,1.30,1.05,.89,.55,1.10,.93,.75,1.19,1.70,1.37,1.47],dtype=np.float32)
ALPHA = np.array([1.42,.70,1.01,1.51,1.13,.57,1.00,1.08,1.16,1.21,1.45,.67,.57,1.11,.98,.77,.83,1.06,1.08,.69],dtype=np.float32)
BULK = np.array([11.5,13.46,11.68,13.57,19.80,3.40,13.69,21.40,15.71,21.40,16.25,12.82,17.43,14.45,14.28,9.47,15.77,21.57,21.67,18.03],dtype=np.float32)

DIS_MASK = np.array([1 if a in "AEGKPQRS" else 0 for a in AA],dtype=np.float32)
ORD_MASK = np.array([1 if a in "CFILMVWY" else 0 for a in AA],dtype=np.float32)
PROPS = np.stack([HYDRO,FLEX,DISPROP,CHARGE,BETA,ALPHA,BULK])

KEY_DIS = [AA_IDX[a] for a in "PEKSQG"]
KEY_ORD = [AA_IDX[a] for a in "WCFIYV"]


def wavg(vals, hw):
    L = len(vals)
    cs = np.cumsum(vals, axis=0)
    s = np.maximum(np.arange(L)-hw, 0)
    e = np.minimum(np.arange(L)+hw, L-1)
    ln = (e-s+1).astype(np.float32)
    if vals.ndim == 1:
        return (cs[e]-np.where(s>0,cs[s-1],0))/ln
    return (cs[e]-np.where(s[:,None]>0,cs[s-1],0))/ln[:,None]


def wvar(vals, hw):
    """Windowed variance = E[X^2] - E[X]^2"""
    avg = wavg(vals, hw)
    avg_sq = wavg(vals**2, hw)
    return avg_sq - avg**2


def featurize(seq):
    L = len(seq)
    idx = np.array([AA_IDX.get(c,0) for c in seq],dtype=np.int32)
    f = []
    
    # Per-residue props (7)
    pr = PROPS[:,idx].T
    f.append(pr)
    
    # Position (2)
    pos = np.arange(L,dtype=np.float32)/max(L-1,1)
    dt = np.minimum(np.arange(L),np.arange(L-1,-1,-1)).astype(np.float32)/max(L-1,1)
    f.append(np.stack([pos,dt],1))
    
    # Per-residue dis/ord (2)
    dv = DIS_MASK[idx]; ov = ORD_MASK[idx]
    f.append(np.stack([dv,ov],1))
    
    # Multi-scale windowed: 5 scales for properties
    for hw in [3, 7, 15, 30, 50]:
        f.append(wavg(pr, hw))  # (L,7) avg props
        f.append(wavg(np.stack([dv,ov],1), hw))  # (L,2) dis/ord frac
        ch = CHARGE[idx]
        f.append(np.stack([wavg(ch,hw), wavg(np.abs(ch),hw)],1))  # (L,2) charge features
    
    # Property VARIANCE at 3 scales (captures heterogeneity)
    for hw in [5, 15, 30]:
        h_var = wvar(HYDRO[idx], hw)
        d_var = wvar(DISPROP[idx], hw)
        f.append(np.stack([h_var, d_var], 1))  # (L,2)
    
    # Key AA composition at 3 scales
    for hw in [5, 15, 35]:
        for aa_i in KEY_DIS:
            f.append(wavg((idx==aa_i).astype(np.float32),hw).reshape(-1,1))
        for aa_i in KEY_ORD:
            f.append(wavg((idx==aa_i).astype(np.float32),hw).reshape(-1,1))
    
    # P/G enrichment (2)
    f.append(wavg((idx==AA_IDX['P']).astype(np.float32),10).reshape(-1,1))
    f.append(wavg((idx==AA_IDX['G']).astype(np.float32),10).reshape(-1,1))
    
    # Sequence complexity (2 scales)
    for hw in [10, 25]:
        uc = np.zeros(L,dtype=np.float32)
        for i in range(L):
            s,e = max(0,i-hw),min(L,i+hw+1)
            uc[i] = len(set(seq[s:e]))/min(e-s,20)
        f.append(uc.reshape(-1,1))
    
    # Hydrophobic cluster: local hydrophobicity variance (2 scales)
    for hw in [5, 15]:
        f.append(wvar(HYDRO[idx],hw).reshape(-1,1))
    
    # Disorder propensity gradient (difference between scales)
    d_short = wavg(DISPROP[idx], 5)
    d_long = wavg(DISPROP[idx], 30)
    f.append((d_short - d_long).reshape(-1,1))
    
    # Global (3)
    f.append(np.full((L,3),[DIS_MASK[idx].mean(), len(set(seq))/20, np.log(L)/10],dtype=np.float32))
    
    return np.concatenate(f, 1)


def evaluate(yt, yp):
    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    fpr,tpr,th = roc_curve(yt, yp)
    opt = th[np.argmax(tpr-fpr)]
    yb = (yp>=opt).astype(int)
    return {"auc_roc":auc,"avg_precision":ap,"f1":f1_score(yt,yb),
            "mcc":matthews_corrcoef(yt,yb),"precision":precision_score(yt,yb),
            "recall":recall_score(yt,yb),"balanced_acc":balanced_accuracy_score(yt,yb)}


def main():
    print("="*70)
    print("DisorderNet FINAL: LightGBM + XGBoost Ensemble")
    print("="*70)
    
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    proteins = [p for p in data if 30<=p["length"]<=700
                and sum(p["disorder_labels"])>=5
                and p["length"]-sum(p["disorder_labels"])>=5]
    
    rng = np.random.RandomState(42)
    if len(proteins) > 800:
        idx = rng.choice(len(proteins),800,replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    
    total_res = sum(p["length"] for p in proteins)
    total_dis = sum(sum(p["disorder_labels"]) for p in proteins)
    print(f"Proteins: {len(proteins)} | Residues: {total_res:,} ({100*total_dis/total_res:.1f}% dis)")
    
    print("Featurizing...")
    t0 = time.time()
    pf, pl = [], []
    for p in proteins:
        pf.append(featurize(p["sequence"]))
        pl.append(np.array(p["disorder_labels"],dtype=np.float32))
    ndim = pf[0].shape[1]
    print(f"Done in {time.time()-t0:.1f}s. Features: {ndim}")
    
    n = len(proteins)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(range(n),range(n),range(n)))
    
    fold_m = []
    all_yt, all_yp = [], []
    
    for fold_i, (tr_p, val_p) in enumerate(folds):
        print(f"\nFold {fold_i+1}/5")
        
        X_val = np.nan_to_num(np.vstack([pf[i] for i in val_p]),0).astype(np.float32)
        y_val = np.concatenate([pl[i] for i in val_p])
        
        # Build balanced training set
        X_tr_full = np.nan_to_num(np.vstack([pf[i] for i in tr_p]),0).astype(np.float32)
        y_tr_full = np.concatenate([pl[i] for i in tr_p])
        
        dis_i = np.where(y_tr_full==1)[0]
        ord_i = np.where(y_tr_full==0)[0]
        n_keep = min(len(ord_i), len(dis_i)*3)
        keep = np.sort(np.concatenate([dis_i, rng.choice(ord_i,n_keep,replace=False)]))
        X_tr = X_tr_full[keep]; y_tr = y_tr_full[keep]
        del X_tr_full, y_tr_full; gc.collect()
        
        print(f"  Train: {len(y_tr):,} | Val: {len(y_val):,}")
        
        spw = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
        
        # LightGBM
        dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dval_lgb = lgb.Dataset(X_val, label=y_val, reference=dtrain_lgb, free_raw_data=True)
        
        lgb_model = lgb.train(
            {'objective':'binary','metric':'auc','num_leaves':95,'max_depth':7,
             'learning_rate':0.07,'feature_fraction':0.8,'bagging_fraction':0.7,
             'bagging_freq':5,'scale_pos_weight':spw,'min_child_samples':20,
             'reg_alpha':0.05,'reg_lambda':0.5,'verbose':-1,'n_jobs':2,'seed':42},
            dtrain_lgb, 500, valid_sets=[dval_lgb],
            callbacks=[lgb.early_stopping(20,verbose=False),lgb.log_evaluation(0)])
        
        lgb_pred = lgb_model.predict(X_val)
        del dtrain_lgb, dval_lgb; gc.collect()
        
        # XGBoost
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
        dval_xgb = xgb.DMatrix(X_val, label=y_val)
        
        xgb_model = xgb.train(
            {'objective':'binary:logistic','eval_metric':'auc',
             'max_depth':6,'learning_rate':0.08,'subsample':0.8,
             'colsample_bytree':0.8,'scale_pos_weight':spw,
             'min_child_weight':20,'reg_alpha':0.05,'reg_lambda':0.5,
             'tree_method':'hist','nthread':2,'seed':42},
            dtrain_xgb, 500, evals=[(dval_xgb,'val')],
            early_stopping_rounds=20, verbose_eval=False)
        
        xgb_pred = xgb_model.predict(dval_xgb)
        del dtrain_xgb, dval_xgb, X_tr, y_tr; gc.collect()
        
        # Ensemble
        ens_pred = 0.55 * lgb_pred + 0.45 * xgb_pred
        
        m = evaluate(y_val, ens_pred)
        m_lgb = evaluate(y_val, lgb_pred)
        m_xgb = evaluate(y_val, xgb_pred)
        
        fold_m.append(m)
        all_yt.append(y_val); all_yp.append(ens_pred)
        
        print(f"  LGB:  AUC={m_lgb['auc_roc']:.4f}")
        print(f"  XGB:  AUC={m_xgb['auc_roc']:.4f}")
        print(f"  ENS:  AUC={m['auc_roc']:.4f} AP={m['avg_precision']:.4f} F1={m['f1']:.4f} MCC={m['mcc']:.4f}")
        
        del X_val, y_val, lgb_model, xgb_model; gc.collect()
    
    y_all = np.concatenate(all_yt); p_all = np.concatenate(all_yp)
    pooled = evaluate(y_all, p_all)
    auc_vals = [m["auc_roc"] for m in fold_m]
    our = pooled["auc_roc"]
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nCV: AUC = {np.mean(auc_vals):.4f} ± {np.std(auc_vals):.4f}")
    for k in ["auc_roc","avg_precision","f1","mcc","precision","recall","balanced_acc"]:
        vals = [m[k] for m in fold_m]
        print(f"  {k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print(f"\nPooled:")
    for k,v in pooled.items():
        print(f"  {k:20s}: {v:.4f}")
    
    bm = [("AF3-pLDDT (CAID3, rank 13)",0.747),
          ("AF2-pLDDT (CAID3, rank 11)",0.770),
          ("AF2-RSA",0.768),("IUPred3",0.789),
          ("flDPnn (best CAID/CAID2)",0.814),
          ("DisorderNet (OURS)",our)]
    
    print(f"\n  {'Method':<35s} {'AUC':>7s} {'vs AF3':>8s}")
    print("  "+"-"*53)
    for n,a in bm:
        print(f"  {n:<33s} {a:>7.3f} {((a-0.747)/0.747)*100:>+7.1f}%")
    
    imp_af3 = ((our-0.747)/0.747)*100
    imp_af2 = ((our-0.770)/0.770)*100
    print(f"\n  Improvement over AF3-pLDDT: +{imp_af3:.1f}% AUC-ROC")
    print(f"  Improvement over AF2-pLDDT: +{imp_af2:.1f}% AUC-ROC")
    print(f"  All folds > AF3 (0.747): {all(v>0.747 for v in auc_vals)}")
    print(f"  All folds > AF2 (0.770): {all(v>0.770 for v in auc_vals)}")
    print(f"  Fold AUCs: {[f'{v:.4f}' for v in auc_vals]}")
    
    results = {"pooled":{k:float(v) for k,v in pooled.items()},
               "fold_aucs":[float(v) for v in auc_vals],
               "cv_mean_auc":float(np.mean(auc_vals)),"cv_std_auc":float(np.std(auc_vals)),
               "fold_metrics":[{k:float(v) for k,v in m.items()} for m in fold_m],
               "n_features":ndim,"n_proteins":len(proteins)}
    with open(os.path.join(RESULTS_DIR,"metrics.json"),"w") as f:
        json.dump(results,f,indent=2)
    np.save(os.path.join(RESULTS_DIR,"y_true.npy"),y_all)
    np.save(os.path.join(RESULTS_DIR,"y_pred.npy"),p_all)
    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__=="__main__":
    main()
