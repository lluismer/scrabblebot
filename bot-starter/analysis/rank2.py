# Wider window: last 7 days, >=4 real bots; place re-ranked among the real bots only
# (sims removed), plus how much of each bot's sample had sims at the table.
import json, math, sys
from datetime import datetime
from collections import defaultdict
ts=lambda s: datetime.fromisoformat(s.replace('Z','')).timestamp()
days=float(sys.argv[1]) if len(sys.argv)>1 else 7
ms=[json.loads(l) for l in open('data/matches.jsonl')]
last=max(ts(m['endedAt']) for m in ms); cut=last-days*86400
st=defaultdict(list); nm={}; simfrac=defaultdict(list); sc=defaultdict(list); bt=defaultdict(list)
for m in ms:
  if ts(m['endedAt'])<cut: continue
  R=[s for s in m['seats'] if not s['sim']]
  if len(R)<4: continue
  hassim=len(R)<len(m['seats'])
  for s in R:
    rp=1+sum(o['score']>s['score'] for o in R)
    # fraction of real opponents beaten (ties half): comparable across 4-6 real seats
    fb=sum((o['score']<s['score'])+0.5*(o['score']==s['score']) for o in R if o is not s)/(len(R)-1)
    st[s['botId']].append(rp); bt[s['botId']].append(fb); nm[s['botId']]=s['name']; simfrac[s['botId']].append(hassim); sc[s['botId']].append(s['score'])
rows=[]
for b,p in st.items():
  n=len(p)
  if n<40: continue
  f=bt[b]; mu=sum(f)/n; se=math.sqrt(sum((x-mu)**2 for x in f)/(n-1)/n)
  rows.append((-mu,se,sum(p)/n,n,nm[b],b,sum(simfrac[b])/n,sum(sc[b])/n))
rows.sort()
print(f"window {days}d; rank by share of real opponents beaten; place = among real seats only")
for i,r in enumerate(rows): print(f"{i+1:2} {r[4][:22]:22} {r[5]:>4} beat={-r[0]:.3f}±{r[1]:.3f} realplace={r[2]:.2f} n={r[3]:4} score={r[7]:6.1f} simTables={r[6]:.0%}")
