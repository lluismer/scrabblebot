# Tobit bid fits + word policy + per-match stats from analysis/fit_real.pkl
import pickle, json, numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
res=pickle.load(open('analysis/fit_real.pkl','rb'))
FEAT=["const","value","vowel","have_it","rack_size","match_frac","balance/10","tiles_won","len_gain","makes_6","makes_7+"]
def X_of(R):
  v,vo,have,rs,vf,fr,bag,bal,won,b0,b1,gain=[R[:,j] for j in range(12)]
  return np.column_stack([np.ones(len(R)),v,vo,np.minimum(have,2),rs,fr,bal/10,won,gain,((b1==6)&(gain>0)),((b1>=7)&(gain>0))]).astype(float)
def tobit(X,y,c,w):
  def nll(p):
    b=p[:-1]; s=np.exp(p[-1]); mu=X@b
    ll=np.where(c==0, norm.logpdf(y,mu,s), norm.logcdf((y+0.5-mu)/s))
    return -(w*ll).sum()
  p0=np.r_[np.linalg.lstsq(X[c==0],y[c==0],rcond=None)[0],np.log(6)]
  r=minimize(nll,p0,method='L-BFGS-B')
  H=r.hess_inv.todense() if hasattr(r.hess_inv,'todense') else r.hess_inv
  return r.x, np.sqrt(np.diag(H))
out={}
for name,R,words,per in res:
  if len(R)==0: continue
  tmax=R[:,14].max(); w=0.5**((tmax-R[:,14])/(2*86400))
  X=X_of(R); y=R[:,12]; c=R[:,13]
  p,se=tobit(X,y,c,w)
  ols=np.linalg.lstsq(X[c==0],y[c==0],rcond=None)[0]
  P=len(per); won=np.mean([q['won'] for q in per]); paid=sum(q['paid'] for q in per)/max(1,sum(q['won'] for q in per))
  sc=np.mean([q['score'] for q in per]); curve=np.mean([q['curve'] for q in per if len(q['curve'])==4],0)
  L=np.array([x['len'] for x in words]); fr=np.array([x['frac'] for x in words]); bal=np.array([x['bal'] for x in words]); rk=np.array([x['rack'] for x in words])
  short=L<5
  o=dict(matches=P,place=np.mean([q['place'] for q in per]),won=won,price=paid,score=sc,curve=curve.round(0).tolist(),
    words_pm=len(L)/P,len=L.mean(),short=short.mean(),bingo7=(L>=7).mean(),
    short_lowbal=(short&(bal<20)).sum()/max(1,(bal<20).sum()),first_word_frac=np.median([x['frac'] for x in words]) if words else 0,
    rack_at_play=rk.mean(),bal_at_play=bal.mean(),end_words=np.mean([x['end'] for x in words]),endrack=np.mean([q['endrack'] for q in per]),
    endbal=np.mean([q['endbal'] for q in per]),sigma=float(np.exp(p[-1])),tobit=dict(zip(FEAT,np.round(p[:-1],2).tolist())),
    z=dict(zip(FEAT,np.round(p[:-1]/se[:-1],1).tolist())),ols=dict(zip(FEAT,np.round(ols,2).tolist())),
    winbids=float(y[c==0].mean()) if (c==0).any() else 0, pred_mean=float((X@p[:-1]).mean()), n_auct=len(R))
  # bid level by phase / balance band
  fr_=R[:,5]; mu=X@p[:-1]
  o['bid_by_third']=[round(float(mu[(fr_>=a)&(fr_<a+1/3)].mean()),1) for a in (0,1/3,2/3)]
  o['bid_by_value']={v:round(float(mu[R[:,0]==v].mean()),1) for v in (1,2,3,4,8,10) if (R[:,0]==v).any()}
  # word length histogram and timing by length
  o['len_hist']={int(k):round(float((L==k).mean()),2) for k in range(2,10) if (L==k).any()}
  o['play_frac_by_len']={int(k):round(float(fr[L==k].mean()),2) for k in range(3,10) if (L==k).sum()>5}
  out[name]=o
def conv(x):
  if isinstance(x,dict): return {k:conv(v) for k,v in x.items()}
  if isinstance(x,list): return [conv(v) for v in x]
  if isinstance(x,(np.floating,np.integer)): return float(x)
  return x
json.dump(conv(out),open('analysis/cards_real.json','w'),indent=1)
for k,o in out.items():
  print(f"\n### {k}  n={o['matches']} place={o['place']:.2f} won={o['won']:.1f} price={o['price']:.1f} score={o['score']:.0f} curve={o['curve']}")
  print(f" words/m={o['words_pm']:.1f} len={o['len']:.2f} short={o['short']:.0%} 7+={o['bingo7']:.0%} medianPlayAt={o['first_word_frac']:.2f} rackAtPlay={o['rack_at_play']:.1f} balAtPlay={o['bal_at_play']:.0f} endPlays={o['end_words']:.0%} endRack={o['endrack']:.1f} endBal={o['endbal']:.0f} shortWhenBal<20={o['short_lowbal']:.0%}")
  print(f" lenhist={o['len_hist']} playFracByLen={o['play_frac_by_len']}")
  print(f" tobit σ={o['sigma']:.1f} meanLatent={o['pred_mean']:.1f} (avg winning bid {o['winbids']:.1f}) byThird={o['bid_by_third']} byValue={o['bid_by_value']}")
  print("  coef", o['tobit']); print("  z   ", o['z']); print("  OLS-on-winners", o['ols'])
