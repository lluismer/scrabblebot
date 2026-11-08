# Real-only reference stats (all-real 6-seat tables) per bot: place, score, tiles, price, word length.
import json, math
from collections import defaultdict
S=defaultdict(lambda: defaultdict(list)); nmatch=0
for l in open('data/matches.jsonl'):
  m=json.loads(l)
  if m['endedAt']<'2026-11-07T19:30' or len(m['seats'])!=6 or any(s['sim'] for s in m['seats']): continue
  nmatch+=1
  for s in m['seats']:
    b=s['botId']; d=S[s['name']]
    won=[a for a in m['auctions'] if a['winner']==b]
    d['place'].append(s['place']); d['score'].append(s['score']); d['won'].append(len(won)); d['paid'].append(sum(a['paid'] for a in won))
    d['wl']+= [len(w['word']) for w in m['words'] if w['botId']==b]
print("all-real 6-seat matches since 19:30:",nmatch)
rows=sorted(((sum(d['place'])/len(d['place']),k,d) for k,d in S.items() if len(d['place'])>=20))
for p,k,d in rows:
  n=len(d['place']); sd=math.sqrt(sum((x-p)**2 for x in d['place'])/(n-1))
  print(f"{k[:20]:20} n={n:3} place={p:.2f}±{sd/math.sqrt(n):.2f} score={sum(d['score'])/n:6.1f} tiles={sum(d['won'])/n:5.1f} price={sum(d['paid'])/max(1,sum(d['won'])):5.1f} wordlen={sum(d['wl'])/max(1,len(d['wl'])):.2f}")
