import json, glob, re, os
ROOT='/users/joeunyook/logit-probe'
ANS=re.compile(r'Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}', re.I)
def parse(t):
    m=list(ANS.finditer(t or ''))
    return (m[-1].group(1) or m[-1].group(2)).upper() if m else None
MODELS=['gemma2b','llama','phi','qwen','dsqwen7','qwen27','gemma31','llama70','dsllama70']
GAMES=['stag_hunt','chicken','bos','pennies']
OPPVALS={'stag_hunt':('STAG','HARE'),'chicken':('YIELD','DARE'),
         'bos':('BACH','STRAVINSKY'),'pennies':('HEADS','TAILS')}
rows=[]
for model in MODELS:
    for env in GAMES:
        for f in sorted(glob.glob(f'{ROOT}/P3_postcot/responses/{model}_{env}_seed*.json')):
            seed=int(re.search(r'seed(\d+)',f).group(1))
            d=json.load(open(f)); s=d['session']
            res=f'{ROOT}/P3_postcot/results/{model}_{env}_seed{seed}.json'
            hist=json.load(open(res)).get('history_records',[]) if os.path.exists(res) else []
            for r in d['rounds']:
                o=r['oracle_p_act1']
                if o is None or abs(o-0.5)<=0.10: continue
                L=parse(r.get('cot_text'))
                if L is None: continue
                wrong=(L=='B')!=(o>0.5)
                h=hist[:r['t']-1]
                v0,v1=OPPVALS[env]
                counts=dict(opp0=sum(1 for e in h if e.get('opp')==v0),
                            opp1=sum(1 for e in h if e.get('opp')==v1),
                            my0=sum(1 for e in h if e.get('my')==v0),
                            my1=sum(1 for e in h if e.get('my')==v1))
                rows.append(dict(model=model,env=env,seed=seed,t=r['t'],oracle=o,
                    hp=s.get('hidden_param'),a0=s.get('act0'),a1=s.get('act1'),
                    letter=L,wrong=wrong,n_prev=len(h),counts=counts,
                    opp_last=h[-1].get('opp') if h else None,
                    my_last=h[-1].get('my') if h else None,
                    cot=r['cot_text']))
json.dump(rows,open('/tmp/claude-2770/-users-joeunyook/04be0756-aaa6-4e4b-af98-e5ce44ec8e84/scratchpad/all2.json','w'))
print('rounds total',len(rows),'wrong',sum(r['wrong'] for r in rows))
