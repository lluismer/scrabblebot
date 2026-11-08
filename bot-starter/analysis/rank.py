# Step 1: rank real bots on real-only tables (no sim seats, >=4 seats) from the last 48h.
import json, pickle, math, sys
from datetime import datetime, timedelta
from collections import defaultdict
ts=lambda s: datetime.fromisoformat(s.replace('Z','')).timestamp()
real=[]; allst=defaultdict(lambda:[0,0.0,0.0]); last=0
with open('data/matches.jsonl') as f:
  for l in f:
    m=json.loads(l); e=ts(m['endedAt']); last=max(last,e)
    for s in m['seats']:
      if not s['sim']: a=allst[s['botId']]; a[0]+=1; a[1]+=s['place']; a[2]+=s['score']
    if any(s['sim'] for s in m['seats']) or len(m['seats'])<4: continue
    m['t']=e; real.append(m)
cut=last-48*3600
recent=[m for m in real if m['t']>=cut]
pickle.dump(recent,open('analysis/recent.pkl','wb'))
st=defaultdict(list); nm={}; sc=defaultdict(list); ns=defaultdict(list)
for m in recent:
  for s in m['seats']:
    st[s['botId']].append(s['place']); sc[s['botId']].append(s['score']); nm[s['botId']]=s['name']; ns[s['botId']].append(len(m['seats']))
rows=[]
for b,p in st.items():
  if len(p)<40: continue
  n=len(p); mu=sum(p)/n; sd=math.sqrt(sum((x-mu)**2 for x in p)/(n-1))
  a=allst[b]
  rows.append((mu,sd/math.sqrt(n),n,b,nm[b],sum(sc[b])/n,sum(ns[b])/n,a[1]/a[0],a[2]/a[0],a[0]))
rows.sort()
print(f"{len(recent)} real-only matches in last 48h (of {len(real)} all-time)")
print("rank bot id place±SE n score seats | all-tables place score n")
for i,r in enumerate(rows):
  print(f"{i+1:2} {r[4][:22]:22} {r[3]:>4} {r[0]:.2f}±{r[1]:.2f} {r[2]:4} {r[5]:6.1f} {r[6]:.1f} | {r[7]:.2f} {r[8]:6.1f} {r[9]}")
