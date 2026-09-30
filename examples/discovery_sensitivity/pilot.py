"""Synthetic resonance search: fixed classifiers, validation-only cuts, profiled nuisances."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS=np.linspace(80,170,16)
NORM=(0.0,0.05,0.15,0.30)
SHAPE=(0.0,0.15,0.35,0.60)

def generate(seed,n=16000):
    rng=np.random.default_rng(seed)
    y=np.repeat([0,1],n)
    # Truncated smooth continuum and a Gaussian resonance, in illustrative GeV units.
    u=rng.random(n); mass_b=80-np.log(1-u*(1-np.exp(-90/65)))*65
    mass_s=rng.normal(125,3,n)
    mass=np.r_[mass_b,mass_s]
    x1=rng.normal(1.0*y,1,len(y))
    # A reconstructed feature correlated with continuum mass; no explicit mass input.
    x2=rng.normal(np.where(y==0,(mass-110)/30,1.2),0.8)
    x3=rng.normal(0,1,len(y)); x4=rng.normal(0,1,len(y))
    # Nonlinear correlation with a synthetic angular interpretation.
    x4[y==1]=0.75*x3[y==1]+rng.normal(0,0.55,n)
    return np.c_[x1,x2,x3,x4],mass,y

def templates(mass,y,score,cut):
    keep=score>=cut
    # Per-class generation statistics do not define physical class priors.
    s=np.histogram(mass[keep&(y==1)],BINS)[0]*180/np.sum(y==1)
    b=np.histogram(mass[keep&(y==0)],BINS)[0]*12000/np.sum(y==0)
    return s,b

def significance(s,b,norm=0.,shape=0.):
    s=np.asarray(s,float); b=np.asarray(b,float)
    if np.any(b<=0): raise ValueError('Zero background bin: merge bins or increase MC, do not add pseudocounts.')
    n=s+b
    if norm==0 and shape==0:
        return float(np.sqrt(2*np.sum(xlogy(n,n/b)-s)))
    centers=(BINS[:-1]+BINS[1:])/2
    tilt=(centers-125)/45
    widths=np.array([norm,shape]); active=widths>0
    def predicted(z):
        theta=np.zeros(2); theta[active]=z
        # Shape-only exponential tilt; selected total yield is preserved.
        q=b*np.exp(theta[1]*shape*tilt); q*=b.sum()/q.sum()
        return q*np.exp(theta[0]*np.log1p(norm))
    def dev(z):
        q=predicted(z)
        return 2*np.sum(q-n+xlogy(n,n/q))+np.dot(z,z)
    starts=[np.zeros(active.sum()),np.ones(active.sum())]
    fits=[minimize(dev,z,method='BFGS',options={'gtol':1e-6}) for z in starts]
    fit=min(fits,key=lambda f:f.fun)
    # Some BFGS precision-loss statuses are harmless; verify stationarity explicitly.
    if not np.isfinite(fit.fun) or np.linalg.norm(fit.jac)>1e-3:
        raise RuntimeError(f'Unreliable nuisance fit: {fit.message}, gradient={fit.jac}')
    # On nominal s+b Asimov data, the unrestricted fit at mu=1, theta=0 has deviance zero.
    return float(np.sqrt(max(0,fit.fun)))

def run(seeds,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); rows=[]; diag=[]
    for seed in seeds:
        train=generate(seed*10+1);val=generate(seed*10+2);test=generate(seed*10+3)
        models={'logistic':LogisticRegression(max_iter=500),'boosted':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=12,l2_regularization=2,random_state=seed)}
        for name,model in models.items():
            model.fit(train[0],train[2]);vs=model.predict_proba(val[0])[:,1];ts=model.predict_proba(test[0])[:,1]
            # Same prespecified quantile menu for each model. Optimize only nominal validation Z.
            cuts=np.quantile(vs[val[2]==1],np.linspace(0,.8,17))
            candidates=[]
            for cut in cuts:
                s,b=templates(val[1],val[2],vs,cut)
                if np.min(b)>=10: candidates.append((significance(s,b),float(cut)))
            _,cut=max(candidates)
            s,b=templates(test[1],test[2],ts,cut)
            diag.append({'seed':seed,'model':name,'threshold':cut,'mass_bins':BINS.tolist(),'s':s.tolist(),'b':b.tolist()})
            for norm in NORM:
                for shape in SHAPE:
                    rows.append(dict(seed=seed,model=name,auc=roc_auc_score(test[2],ts),threshold=cut,norm=norm,shape=shape,z=significance(s,b,norm,shape),s=s.sum(),b=b.sum()))
        print('completed seed',seed,flush=True)
    with open(out/'runs.csv','w') as f:
        wr=csv.DictWriter(f,rows[0].keys());wr.writeheader();wr.writerows(rows)
    (out/'templates.json').write_text(json.dumps(diag,indent=2))
    summaries=[]
    for norm in NORM:
        for shape in SHAPE:
            differences=[]; reversals=[]
            for seed in seeds:
                a=next(r for r in rows if r['seed']==seed and r['model']=='logistic' and r['norm']==norm and r['shape']==shape)
                b=next(r for r in rows if r['seed']==seed and r['model']=='boosted' and r['norm']==norm and r['shape']==shape)
                differences.append(b['z']-a['z']);reversals.append((b['auc']-a['auc'])*(b['z']-a['z'])<0)
            summaries.append(dict(norm=norm,shape=shape,mean_delta_z=np.mean(differences),seed_sd_delta_z=np.std(differences,ddof=1) if len(seeds)>1 else 0,reversal_count=int(sum(reversals)),seeds=len(seeds)))
    with open(out/'summary.csv','w') as f:
        wr=csv.DictWriter(f,summaries[0].keys());wr.writeheader();wr.writerows(summaries)
    plt.rcParams.update({'figure.facecolor':'#f4eee2','axes.facecolor':'#f4eee2','text.color':'#36213f','axes.labelcolor':'#36213f','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,color in [('logistic','#907a9c'),('boosted','#36213f')]:
        subset=[r for r in rows if r['model']==name and r['norm']==0 and r['shape']==0]
        axs[0].scatter([r['auc'] for r in subset],[r['z'] for r in subset],label=name,color=color)
    axs[0].set(xlabel='Test ROC AUC',ylabel='Nominal expected local Z',title='Each point is an independent simulation seed');axs[0].legend()
    mat=np.array([r['mean_delta_z'] for r in summaries]).reshape(4,4)
    lim=max(abs(mat.min()),abs(mat.max()),.01)
    im=axs[1].imshow(mat,origin='lower',cmap='PuOr_r',vmin=-lim,vmax=lim)
    axs[1].set(xticks=range(4),xticklabels=SHAPE,yticks=range(4),yticklabels=NORM,xlabel='Shape tilt strength',ylabel='Background normalization uncertainty',title='Mean Z(boosted) minus Z(logistic)')
    for i in range(4):
        for j in range(4): axs[1].text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',color='black')
    fig.colorbar(im,ax=axs[1],label='Difference in expected local Z');fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    report={'seeds':seeds,'auc_means':{name:float(np.mean([r['auc'] for r in rows if r['model']==name and r['norm']==0 and r['shape']==0])) for name in models},'reversal_scenarios':sum(r['reversal_count']>0 for r in summaries),'total_scenarios':len(summaries)}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);p.add_argument('--out',default='results');a=p.parse_args();run(a.seeds,a.out)
