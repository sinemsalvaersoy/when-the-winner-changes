"""Paired event-level synthetic detector response, frozen classifiers and selection cuts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from pilot import generate,templates,significance,BINS,plt

VARIATIONS={'nominal':(0.,1.),'scale_down':(-.01,1.),'scale_up':(.01,1.),'resolution_down':(0.,.9),'resolution_up':(0.,1.1)}

def response(latent,noise_seed,scale=0.,resolution=1.):
    # No class label is read: apply the identical response law to every event.
    x,m,y=latent;rng=np.random.default_rng(noise_seed)
    zm=rng.normal(size=len(m));zx=rng.normal(size=x.shape)
    reco_m=m*(1+scale)+1.5*resolution*zm
    reco_x=x+resolution*zx*np.array([.10,.15,.08,.08])
    # x2 is a mass-correlated proxy; the same energy scale acts on it.
    reco_x[:,1]+=scale*m/30
    return reco_x,reco_m,y

def hist_components(m0,mv,y,s0,sv,cut):
    # Factor the change into score selection and reconstructed mass movement.
    return {'nominal':templates(m0,y,s0,cut),'score_only':templates(m0,y,sv,cut),'mass_only':templates(mv,y,s0,cut),'full':templates(mv,y,sv,cut)}

def migration(m0,mv,y,s0,sv,cut):
    a=s0>=cut;b=sv>=cut
    win0=(m0>=BINS[0])&(m0<=BINS[-1]);winv=(mv>=BINS[0])&(mv<=BINS[-1])
    out=[]
    for label,cls in [('background',0),('signal',1)]:
        cls_mask=y==cls;n=int(cls_mask.sum())
        aa=a&cls_mask;bb=b&cls_mask;fa=aa&win0;fb=bb&winv
        out.append(dict(sample=label,n_generated=n,selected_nominal=int(aa.sum()),selected_varied=int(bb.sum()),entered=int((~a&b&cls_mask).sum()),exited=int((a&~b&cls_mask).sum()),final_entered=int((~fa&fb&cls_mask).sum()),final_exited=int((fa&~fb&cls_mask).sum()),mass_bin_changed=int((aa&bb&win0&winv&(np.digitize(m0,BINS)!=np.digitize(mv,BINS))).sum())))
    return out

def morph(b0,bdown,bup,theta):
    # Positive log-linear interpolation within supplied endpoints; no extrapolation.
    if np.any(np.asarray([b0,bdown,bup])<=0):raise ValueError('Insufficient background MC bin support')
    endpoint=bup if theta>=0 else bdown
    return b0*np.exp(abs(theta)*np.log(endpoint/b0))

def response_profile_z(s,b,variations,norm=.15):
    n=s+b
    def objective(z):
        q=morph(b,*variations['scale'],z[0])
        # Independent multiplicative template effects; interaction not modeled.
        q*=morph(b,*variations['resolution'],z[1])/b
        q*=np.exp(z[2]*np.log1p(norm))
        return float(2*np.sum(q-n+xlogy(n,n/q))+np.dot(z,z))
    bounds=[(-1,1),(-1,1),(-8,8)]
    fits=[minimize(objective,z,method='Powell',bounds=bounds,options={'xtol':1e-7,'ftol':1e-9}) for z in [np.zeros(3),np.array([.5,.5,0]),np.array([-.5,-.5,0])]]
    f=min(fits,key=lambda x:x.fun)
    if not f.success or not np.isfinite(f.fun):raise RuntimeError('Response profile fit failed')
    return np.sqrt(max(0,f.fun)),f.x.tolist()

def run(seeds,out,n=32000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];moves=[];profiles=[];stored=[]
    for seed in seeds:
        latents=[generate(seed*10+i,n) for i in [1,2,3]]
        train,val,test=[response(latent,seed*100+i) for i,latent in enumerate(latents)]
        models={'logistic':LogisticRegression(max_iter=500),'boosted':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=12,l2_regularization=2,random_state=seed)}
        for name,model in models.items():
            model.fit(train[0],train[2]);vs=model.predict_proba(val[0])[:,1];ts=model.predict_proba(test[0])[:,1]
            candidates=[]
            for cut in np.quantile(vs[val[2]==1],np.linspace(0,.8,17)):
                s,b=templates(val[1],val[2],vs,cut)
                if np.min(b)>=10:candidates.append((significance(s,b),float(cut)))
            _,cut=max(candidates);s0,b0=templates(test[1],test[2],ts,cut)
            variation_templates={};event_masks={'label':test[2],'nominal_mass':test[1],'nominal_score':ts}
            for variation,(scale,resolution) in VARIATIONS.items():
                varied=response(latents[2],seed*100+2,scale,resolution)
                score=model.predict_proba(varied[0])[:,1]
                event_masks[variation+'_mass']=varied[1];event_masks[variation+'_score']=score
                components=hist_components(test[1],varied[1],test[2],ts,score,cut)
                s,b=components['full'];variation_templates[variation]=(s,b)
                for m in migration(test[1],varied[1],test[2],ts,score,cut):moves.append(dict(seed=seed,model=name,variation=variation,**m))
                rows.append(dict(seed=seed,model=name,variation=variation,threshold=cut,auc=roc_auc_score(test[2],score),signal_yield=s.sum(),background_yield=b.sum(),z_conditional_known_b=significance(s,b),relative_signal_shift=s.sum()/s0.sum()-1,relative_background_shift=b.sum()/b0.sum()-1))
                stored.append(dict(seed=seed,model=name,variation=variation,components={k:{'s':ss.tolist(),'b':bb.tolist()} for k,(ss,bb) in components.items()}))
            np.savez_compressed(out/f'events_{seed}_{name}.npz',**event_masks,threshold=cut)
            syst={'scale':(variation_templates['scale_down'][1],variation_templates['scale_up'][1]),'resolution':(variation_templates['resolution_down'][1],variation_templates['resolution_up'][1])}
            z,theta=response_profile_z(s0,b0,syst)
            znorm=significance(s0,b0,.15,0)
            if z>znorm+1e-5:raise RuntimeError('Profile result violates nested-model bound')
            profiles.append(dict(seed=seed,model=name,z_known_background=significance(s0,b0),z_norm_only=znorm,z_norm_and_response=float(z),bestfit_scale=theta[0],bestfit_resolution=theta[1],bestfit_norm=theta[2],response_at_boundary=bool(max(abs(theta[0]),abs(theta[1]))>.999)))
        print('detector seed',seed,'complete',flush=True)
    for filename,data in [('runs.csv',rows),('migrations.csv',moves),('profiled.csv',profiles)]:
        with (out/filename).open('w') as f:
            wr=csv.DictWriter(f,data[0].keys());wr.writeheader();wr.writerows(data)
    (out/'templates.json').write_text(json.dumps(stored,indent=2))
    # Readable mechanism figure with empirical migration fractions, not invented arrows.
    plt.rcParams.update({'figure.facecolor':'#f4eee2','axes.facecolor':'#f4eee2','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    variants=list(VARIATIONS)[1:]
    for name,color in [('logistic','#957d9e'),('boosted','#36213f')]:
        frac=[np.mean([(m['entered']+m['exited'])/m['n_generated']*100 for m in moves if m['model']==name and m['variation']==v and m['sample']=='background']) for v in variants]
        axs[0].plot(range(4),frac,'o-',label=name,color=color)
        selected=[p for p in profiles if p['model']==name]
        vals=[[p[k] for p in selected] for k in ['z_known_background','z_norm_only','z_norm_and_response']]
        axs[1].errorbar(range(3),np.mean(vals,axis=1),yerr=np.std(vals,axis=1,ddof=1),fmt='o-',capsize=3,color=color,label=name)
    axs[0].set(xticks=range(4),xticklabels=['Scale -1%','Scale +1%','Resolution -10%','Resolution +10%'],ylabel='Background entering or exiting selection (%)',title='Paired events; frozen model and cut');axs[0].tick_params(axis='x',rotation=15)
    axs[1].set(xticks=range(3),xticklabels=['Known background','Normalization','Normalization + response'],ylabel='Nominal expected local Z',title='Mean and simulation seed standard deviation');axs[1].tick_params(axis='x',rotation=15)
    for ax in axs:ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'detector_mechanism.png',dpi=180);plt.close(fig)
    report={'seeds':seeds,'events_per_class_per_split':n,'profile_means':{name:{k:float(np.mean([p[k] for p in profiles if p['model']==name])) for k in ['z_known_background','z_norm_only','z_norm_and_response']} for name in models},'boundary_fits':sum(p['response_at_boundary'] for p in profiles)}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55]);p.add_argument('--out',default='results/detector');p.add_argument('--n',type=int,default=32000);a=p.parse_args();run(a.seeds,a.out,a.n)
