import json,re
from collections import defaultdict
rows=json.load(open('all2.json'))
ACT={'stag_hunt':['STAG','HARE'],'chicken':['YIELD','DARE'],'bos':['BACH','STRAVINSKY'],'pennies':['HEADS','TAILS']}
MATCH={'stag_hunt':True,'bos':True,'pennies':True,'chicken':False}
SUBJ=r'(?:opponent|colleague|other hunter|hunter|they|opp)'
FUT=(rf'{SUBJ}\s*(?:is|are|will|would|is most|are most|is highly|are highly)?[^.]{{0,60}}?'
     r'(?:likely to|will|would|going to|expected to|should)\s+(?:also\s+|again\s+)?'
     r'(?:choose|play|hunt|pick|select|go with)\s*[:\*\s]*(%s)\b')
COND=re.compile(r'\b(if|were|suppose|assuming|in case|unless|whether)\b',re.I)
def predicted(txt,env):
    """last UNCONDITIONAL future-tense prediction, in the final 60% of the text"""
    acts=ACT[env]; cut=int(len(txt)*0.40); best=None
    for a in acts:
        for m in re.finditer(FUT%a,txt,re.I):
            if m.start()<cut: continue
            if COND.search(txt[max(0,m.start()-80):m.start()+10]): continue
            if best is None or m.start()>best[1]: best=(a,m.start())
    return best[0] if best else None
def canon(x,env):
    for a in ACT[env]:
        if x.upper().startswith(a[:4]) or a.startswith(x.upper()[:4]): return a
    return x.upper()
stat=defaultdict(lambda:defaultdict(lambda:[0,0])); ex=[]
for r in rows:
    p=predicted(r['cot'],r['env'])
    if p is None: continue
    ch=canon((r['a1'] if r['letter']=='B' else r['a0']).upper(),r['env'])
    br=p if MATCH[r['env']] else [a for a in ACT[r['env']] if a!=p][0]
    b='wrong' if r['wrong'] else 'correct'
    stat[r['model']][b][0]+=1
    if ch!=br:
        stat[r['model']][b][1]+=1
        if r['wrong'] and len(ex)<250: ex.append(dict(model=r['model'],env=r['env'],seed=r['seed'],
            t=r['t'],hp=r['hp'],pred=p,chose=ch,should=br,oracle=r['oracle'],cot=r['cot']))
print('TYPE 1 -- chosen action contradicts the model\'s OWN final, unconditional prediction')
print('                 FAILED rounds             CORRECT rounds')
print('model          n   inconsist   rate  |    n   inconsist   rate')
T=[0,0,0,0]
for m in ['gemma2b','llama','phi','qwen','dsqwen7','qwen27','gemma31','llama70','dsllama70']:
    w=stat[m]['wrong']; c=stat[m]['correct']
    if w[0]<5: continue
    T[0]+=w[0];T[1]+=w[1];T[2]+=c[0];T[3]+=c[1]
    cr=f'{100*c[1]/c[0]:6.1f}%' if c[0] else '   n/a'
    print(f'{m:12}{w[0]:5}{w[1]:8}{100*w[1]/w[0]:8.1f}% |{c[0]:6}{c[1]:8}{cr}')
print(f'{"TOTAL":12}{T[0]:5}{T[1]:8}{100*T[1]/T[0]:8.1f}% |{T[2]:6}{T[3]:8}{100*T[3]/T[2]:6.1f}%')
json.dump(ex,open('type1_final.json','w'))
