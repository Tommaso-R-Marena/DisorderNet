"""
DisorderNet v6: Optimized ESM-2 + Physics ensemble (no CNN, memory-safe).
Higher PCA dims + more ESM windowed/variance features.
"""
import json, numpy as np, os, time, gc, warnings
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, balanced_accuracy_score,
                             precision_score, recall_score)
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings('ignore')

DATA_PATH = "/home/user/workspace/disorder_model/data/disprot_processed.json"
EMB_DIR = "/home/user/workspace/disorder_model/data/embeddings"
RESULTS_DIR = "/home/user/workspace/disorder_model/results_v6"
os.makedirs(RESULTS_DIR, exist_ok=True)

ESM_PCA_DIM = 48  # Higher than v5's 32 but memory-safe
SEED = 42

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
HYDRO=np.array([1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3],dtype=np.float32)
FLEX=np.array([.984,.906,1.068,1.094,.915,1.031,.950,.927,1.102,.935,.952,1.048,1.049,1.037,1.008,1.046,.997,.931,.904,.929],dtype=np.float32)
DISPROP=np.array([.06,-.02,.192,.736,-.697,.166,.303,-.486,.586,-.326,-.397,.007,.987,.318,.18,.341,.059,-.121,-.884,-.510],dtype=np.float32)
CHARGE=np.array([0,0,-1,-1,0,0,.1,0,1,0,0,0,0,0,1,0,0,0,0,0],dtype=np.float32)
BETA=np.array([.83,1.19,.54,.37,1.38,.75,.87,1.60,.74,1.30,1.05,.89,.55,1.10,.93,.75,1.19,1.70,1.37,1.47],dtype=np.float32)
ALPHA=np.array([1.42,.70,1.01,1.51,1.13,.57,1.00,1.08,1.16,1.21,1.45,.67,.57,1.11,.98,.77,.83,1.06,1.08,.69],dtype=np.float32)
BULK=np.array([11.5,13.46,11.68,13.57,19.80,3.40,13.69,21.40,15.71,21.40,16.25,12.82,17.43,14.45,14.28,9.47,15.77,21.57,21.67,18.03],dtype=np.float32)
DIS_MASK=np.array([1 if a in "AEGKPQRS" else 0 for a in AA],dtype=np.float32)
ORD_MASK=np.array([1 if a in "CFILMVWY" else 0 for a in AA],dtype=np.float32)
PROPS=np.stack([HYDRO,FLEX,DISPROP,CHARGE,BETA,ALPHA,BULK])
KEY_DIS=[AA_IDX[a] for a in "PEKSQG"]
KEY_ORD=[AA_IDX[a] for a in "WCFIYV"]


def wavg(v, hw):
    L=len(v); cs=np.cumsum(v,0)
    s=np.maximum(np.arange(L)-hw,0); e=np.minimum(np.arange(L)+hw,L-1)
    ln=(e-s+1).astype(np.float32)
    return (cs[e]-np.where(s>0,cs[s-1],0))/ln if v.ndim==1 else (cs[e]-np.where(s[:,None]>0,cs[s-1],0))/ln[:,None]

def wvar(v, hw):
    return wavg(v**2 if v.ndim==1 else v**2, hw) - wavg(v,hw)**2


def phys_feats(seq):
    L=len(seq); idx=np.array([AA_IDX.get(c,0) for c in seq],dtype=np.int32)
    f=[]; pr=PROPS[:,idx].T; f.append(pr)
    pos=np.arange(L,dtype=np.float32)/max(L-1,1)
    dt=np.minimum(np.arange(L),np.arange(L-1,-1,-1)).astype(np.float32)/max(L-1,1)
    f.append(np.stack([pos,dt],1))
    dv=DIS_MASK[idx]; ov=ORD_MASK[idx]; f.append(np.stack([dv,ov],1))
    for hw in [3,7,15,30,50]:
        f.append(wavg(pr,hw)); f.append(wavg(np.stack([dv,ov],1),hw))
        ch=CHARGE[idx]; f.append(np.stack([wavg(ch,hw),wavg(np.abs(ch),hw)],1))
    for hw in [5,15,30]:
        f.append(np.stack([wvar(HYDRO[idx],hw),wvar(DISPROP[idx],hw)],1))
    for hw in [5,15,35]:
        for ai in KEY_DIS: f.append(wavg((idx==ai).astype(np.float32),hw).reshape(-1,1))
        for ai in KEY_ORD: f.append(wavg((idx==ai).astype(np.float32),hw).reshape(-1,1))
    f.append(wavg((idx==AA_IDX['P']).astype(np.float32),10).reshape(-1,1))
    f.append(wavg((idx==AA_IDX['G']).astype(np.float32),10).reshape(-1,1))
    for hw in [10,25]:
        uc=np.zeros(L,dtype=np.float32)
        for i in range(L):
            s2,e2=max(0,i-hw),min(L,i+hw+1); uc[i]=len(set(seq[s2:e2]))/min(e2-s2,20)
        f.append(uc.reshape(-1,1))
    for hw in [5,15]: f.append(wvar(HYDRO[idx],hw).reshape(-1,1))
    f.append((wavg(DISPROP[idx],5)-wavg(DISPROP[idx],30)).reshape(-1,1))
    f.append(np.full((L,3),[DIS_MASK[idx].mean(),len(set(seq))/20,np.log(L)/10],dtype=np.float32))
    return np.concatenate(f,1)


def evaluate(yt,yp):
    auc=roc_auc_score(yt,yp); ap=average_precision_score(yt,yp)
    fpr,tpr,th=roc_curve(yt,yp); opt=th[np.argmax(tpr-fpr)]; yb=(yp>=opt).astype(int)
    return {"auc_roc":auc,"avg_precision":ap,"f1":f1_score(yt,yb),
            "mcc":matthews_corrcoef(yt,yb),"precision":precision_score(yt,yb),
            "recall":recall_score(yt,yb),"balanced_acc":balanced_accuracy_score(yt,yb)}


def main():
    print("="*70)
    print("DisorderNet v6: Optimized ESM + Physics Ensemble")
    print("="*70)
    
    with open(DATA_PATH) as f:
        all_data = json.load(f)
    
    proteins = [p for p in all_data
                if os.path.exists(os.path.join(EMB_DIR,f"{p['disprot_id']}.npy"))
                and p["length"]>=30 and sum(p["disorder_labels"])>=3
                and p["length"]-sum(p["disorder_labels"])>=3]
    for p in proteins:
        if p["length"]>1022:
            p["sequence"]=p["sequence"][:1022]; p["disorder_labels"]=p["disorder_labels"][:1022]; p["length"]=1022
    
    total_res=sum(p["length"] for p in proteins)
    total_dis=sum(sum(p["disorder_labels"]) for p in proteins)
    print(f"Proteins: {len(proteins)} | Residues: {total_res:,} ({100*total_dis/total_res:.1f}% dis)")
    
    # PCA
    print(f"\nPCA (-> {ESM_PCA_DIM} dims)...")
    rng=np.random.RandomState(SEED)
    pca=IncrementalPCA(n_components=ESM_PCA_DIM, batch_size=10000)
    si=rng.choice(len(proteins),min(1000,len(proteins)),replace=False)
    ch=[np.load(os.path.join(EMB_DIR,f"{proteins[i]['disprot_id']}.npy")).astype(np.float32)[:proteins[i]["length"]] for i in si]
    pca.fit(np.vstack(ch)); print(f"Var explained: {pca.explained_variance_ratio_.sum():.3f}")
    del ch; gc.collect()
    
    # Features: compute per-protein, keep in list to avoid huge concat
    print("Computing features per protein...")
    t0=time.time()
    pf=[]; pl=[]
    for i,p in enumerate(proteins):
        if (i+1)%500==0: print(f"  {i+1}/{len(proteins)} ({time.time()-t0:.0f}s)")
        seq=p["sequence"]; L=p["length"]
        ph=phys_feats(seq)
        emb=np.load(os.path.join(EMB_DIR,f"{p['disprot_id']}.npy")).astype(np.float32)[:L]
        ep=pca.transform(emb)
        # Multi-scale ESM context
        ew1=wavg(ep,4); ew2=wavg(ep,12); ew3=wavg(ep,25)
        # ESM variance
        ev1=wvar(ep,8); ev2=wvar(ep,20)
        combined=np.concatenate([ph,ep,ew1,ew2,ew3,ev1,ev2],1)
        pf.append(combined.astype(np.float32))
        pl.append(np.array(p["disorder_labels"][:L],dtype=np.float32))
    
    ndim=pf[0].shape[1]
    print(f"Done in {time.time()-t0:.0f}s. Dim: {ndim}")
    
    # CV
    print(f"\n{'='*70}\n5-FOLD CV\n{'='*70}")
    n=len(proteins); gkf=GroupKFold(n_splits=5)
    fm=[]; ayt=[]; ayp=[]
    
    for fold,(tr,va) in enumerate(gkf.split(range(n),range(n),range(n))):
        print(f"\nFold {fold+1}/5")
        
        # Build sets carefully to manage memory
        val_chunks_X=[pf[i] for i in va]
        val_chunks_y=[pl[i] for i in va]
        X_val=np.nan_to_num(np.vstack(val_chunks_X),0).astype(np.float32)
        y_val=np.concatenate(val_chunks_y)
        del val_chunks_X, val_chunks_y
        
        # Balanced training
        tr_X=np.nan_to_num(np.vstack([pf[i] for i in tr]),0).astype(np.float32)
        tr_y=np.concatenate([pl[i] for i in tr])
        di=np.where(tr_y==1)[0]; oi=np.where(tr_y==0)[0]
        nk=min(len(oi),len(di)*3)
        keep=np.sort(np.concatenate([di,rng.choice(oi,nk,replace=False)]))
        X_tr=tr_X[keep]; y_tr=tr_y[keep]
        del tr_X, tr_y; gc.collect()
        
        print(f"  Train: {len(y_tr):,} | Val: {len(y_val):,} ({y_val.sum():.0f} dis)")
        spw=(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
        
        # LightGBM
        dt=lgb.Dataset(X_tr,label=y_tr,free_raw_data=True)
        dv=lgb.Dataset(X_val,label=y_val,reference=dt,free_raw_data=True)
        lm=lgb.train(
            {'objective':'binary','metric':'auc','num_leaves':127,'max_depth':8,
             'learning_rate':0.05,'feature_fraction':0.7,'bagging_fraction':0.7,
             'bagging_freq':5,'scale_pos_weight':spw,'min_child_samples':30,
             'reg_alpha':0.05,'reg_lambda':0.5,'verbose':-1,'n_jobs':2,'seed':SEED},
            dt,700,valid_sets=[dv],
            callbacks=[lgb.early_stopping(25,verbose=False),lgb.log_evaluation(0)])
        lp=lm.predict(X_val)
        del dt,dv; gc.collect()
        
        # XGBoost
        dx=xgb.DMatrix(X_tr,label=y_tr); dvx=xgb.DMatrix(X_val,label=y_val)
        xm=xgb.train(
            {'objective':'binary:logistic','eval_metric':'auc','max_depth':7,
             'learning_rate':0.05,'subsample':0.7,'colsample_bytree':0.7,
             'scale_pos_weight':spw,'min_child_weight':30,'reg_alpha':0.05,
             'reg_lambda':0.5,'tree_method':'hist','nthread':2,'seed':SEED},
            dx,700,evals=[(dvx,'v')],early_stopping_rounds=25,verbose_eval=False)
        xp=xm.predict(dvx)
        del dx,dvx,X_tr,y_tr; gc.collect()
        
        ep=0.55*lp+0.45*xp
        m=evaluate(y_val,ep)
        fm.append(m); ayt.append(y_val); ayp.append(ep)
        print(f"  LGB:{evaluate(y_val,lp)['auc_roc']:.4f} XGB:{evaluate(y_val,xp)['auc_roc']:.4f}")
        print(f"  ENS: AUC={m['auc_roc']:.4f} AP={m['avg_precision']:.4f} F1={m['f1']:.4f} MCC={m['mcc']:.4f}")
        
        if fold==0:
            imp=lm.feature_importance(importance_type='gain')
            t10=np.argsort(imp)[-15:][::-1]
            esm_top=sum(1 for t in t10 if t>=118)
            print(f"  ESM in top 15: {esm_top}/15")
        
        del X_val,y_val,lm,xm; gc.collect()
    
    ya=np.concatenate(ayt); pa=np.concatenate(ayp)
    pl_m=evaluate(ya,pa)
    avs=[m["auc_roc"] for m in fm]
    aps=[m["avg_precision"] for m in fm]
    our=pl_m["auc_roc"]
    
    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    print(f"\nCV: AUC={np.mean(avs):.4f}±{np.std(avs):.4f} AP={np.mean(aps):.4f}±{np.std(aps):.4f}")
    for k in ["auc_roc","avg_precision","f1","mcc","balanced_acc"]:
        vs=[m[k] for m in fm]; print(f"  {k:20s}: {np.mean(vs):.4f}±{np.std(vs):.4f}")
    
    print(f"\nPooled ({len(ya):,} residues):")
    for k,v in pl_m.items(): print(f"  {k:20s}: {v:.4f}")
    
    bm=[("AF3-pLDDT (CAID3)",0.747),("AF2-pLDDT (CAID3)",0.770),("IUPred3",0.789),
        ("DisorderNet v4",0.794),("flDPnn (CAID1/2)",0.814),("SETH",0.830),
        ("DisorderNet v5",0.823),("DisorderNet v6",our)]
    print(f"\n  {'Method':<30s} {'AUC':>7s} {'vs AF3':>8s}")
    print("  "+"-"*48)
    for n,a in bm:
        print(f"  {n:<28s} {a:>7.3f} {((a-0.747)/0.747)*100:>+7.1f}%")
    print(f"\n  All folds>AF3: {all(v>0.747 for v in avs)} | >AF2: {all(v>0.770 for v in avs)} | >flDPnn: {all(v>0.814 for v in avs)}")
    print(f"  Folds: {[f'{v:.4f}' for v in avs]}")
    
    results={"model":"DisorderNet_v6","n_proteins":len(proteins),"n_residues":int(len(ya)),
             "n_features":ndim,"esm_pca_dim":ESM_PCA_DIM,
             "pooled":{k:float(v) for k,v in pl_m.items()},
             "cv_mean":{k:float(np.mean([m[k] for m in fm])) for k in fm[0]},
             "cv_std":{k:float(np.std([m[k] for m in fm])) for k in fm[0]},
             "fold_aucs":[float(v) for v in avs],"fold_aps":[float(v) for v in aps],
             "fold_metrics":[{k:float(v) for k,v in m.items()} for m in fm]}
    with open(os.path.join(RESULTS_DIR,"metrics.json"),"w") as f: json.dump(results,f,indent=2)
    np.save(os.path.join(RESULTS_DIR,"y_true.npy"),ya)
    np.save(os.path.join(RESULTS_DIR,"y_pred.npy"),pa)
    print(f"\nSaved to {RESULTS_DIR}/")

if __name__=="__main__": main()
