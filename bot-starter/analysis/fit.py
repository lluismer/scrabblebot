# Steps 2-4: rebuild each target's public state at every auction, fit a censored
# (Tobit) bid model, and summarise word play. Usage: python analysis/fit.py
import json, math, sys, pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import norm
ts=lambda s: datetime.fromisoformat(s.replace('Z','')).timestamp()
VAL=dict(A=1,B=3,C=3,D=2,E=1,F=4,G=2,H=4,I=1,J=8,K=5,L=1,M=3,N=1,O=1,P=3,Q=10,R=1,S=1,T=1,U=1,V=4,W=4,X=8,Y=4,Z=10)
TARGETS=["LexForge-1107","RackMaw","GodOfThudner","Bron","Lbender","LUCENT","aipom","LexiconWarrior","Odysseus's Voyage","Tow57-Auctioneer"]
CAP=int(sys.argv[1]) if len(sys.argv)>1 else 350
W=None; WL=None
def init():
  global W,WL
  ws=[w.strip() for w in open('../spacetimedb/wordlist.txt') if 3<=len(w.strip())<=9]
  W=np.zeros((len(ws),26),np.int8)
  for i,w in enumerate(ws):
    for c in w: W[i,ord(c)-65]+=1
  WL=np.array([len(w) for w in ws],np.int8)
def bestlen(r):
  ok=(W<=r).all(1)
  return int(WL[ok].max()) if ok.any() else 0
def analyse(name):
  init()
  ms=pickle.load(open('analysis/m7.pkl','rb'))
  mine=[m for m in ms if any(s['name']==name and not s['sim'] for s in m['seats'])][-CAP:]
  rows=[]; words=[]; per=[]
  for m in mine:
    b=[s for s in m['seats'] if s['name']==name][0]['botId']
    n=len(m['auctions']); e=ts(m['endedAt'])
    wk=defaultdict(list)
    for w in m['words']:
      if w['botId']!=b: continue
      k=n-math.ceil((e-ts(w['at']))/1.0); k=min(max(k,0),n)
      wk[k].append(w)
    # hidden start tiles: letters needed beyond what was won
    hid=np.zeros(26,int); rk=np.zeros(26,int)
    for k in range(n+1):
      for w in wk[k]:
        for c in w['word']:
          i=ord(c)-65
          if rk[i]==0: hid[i]+=1
          else: rk[i]-=1
      if k<n and m['auctions'][k]['winner']==b: rk[ord(m['auctions'][k]['letter'])-65]+=1
    rack=hid.copy(); bal=100; won=0; paid=0; score=0; curve=[]; cache={}
    for k in range(n+1):
      for w in wk[k]:
        words.append(dict(len=len(w['word']),k=k,frac=k/n,rack=int(rack.sum()),bal=bal,rew=w['reward'],end=k>=n-1))
        for c in w['word']: rack[ord(c)-65]-=1
        bal+=w['reward']; score+=w['reward']
      if k==n: break
      a=m['auctions'][k]; L=a['letter']; i=ord(L)-65
      key=rack.tobytes()
      if key not in cache: cache[key]=bestlen(rack)
      b0=cache[key]; r2=rack.copy(); r2[i]+=1; b1=bestlen(r2)
      nv=int(sum(rack[ord(c)-65] for c in 'AEIOU')); rs=int(rack.sum())
      if a['winner']==b: y,cens=a['topBid'],0
      elif a['winner'] is None: y,cens=0,1
      else: y,cens=a['paid'],1
      if y is not None:
        rows.append([VAL[L],L in 'AEIOU',int(rack[i]),rs,nv/max(rs,1),k/n,n-k-1,bal,won,b0,b1,max(b1-b0,0),y,cens,m['t']])
      if a['winner']==b:
        rack[i]+=1; won+=1; paid+=a['paid']; bal-=a['paid']
      if k in (n//4,n//2,3*n//4): curve.append(score)
    curve.append(score)
    place=[s for s in m['seats'] if s['botId']==b][0]['place']
    per.append(dict(won=won,paid=paid,score=score,curve=curve,endrack=int(rack.sum()),endbal=bal,place=place))
  return name,np.array(rows,float),words,per
if __name__=='__main__':
  ms=[json.loads(l) for l in open('data/matches.jsonl')]
  last=max(ts(m['endedAt']) for m in ms)
  keep=[]
  for m in ms:
    m['t']=ts(m['endedAt'])
    if m['endedAt']>='2026-11-07T19:30' and len(m['seats'])==6 and not any(s['sim'] for s in m['seats']): keep.append(m)
  pickle.dump(keep,open('analysis/m7.pkl','wb'))
  with Pool(len(TARGETS)) as p: res=p.map(analyse,TARGETS)
  pickle.dump(res,open('analysis/fit_real.pkl','wb')); print("ok",[(r[0],len(r[1])) for r in res])
