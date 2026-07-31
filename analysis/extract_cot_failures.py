import json, glob, re, os
from collections import Counter, defaultdict
ROOT='/users/joeunyook/logit-probe'
ANS=re.compile(r'Answer\s*:\s*([AB])\b|\\boxed\{([AB])\}', re.I)
def parse(t):
    m=list(ANS.finditer(t or ''))
    return (m[-1].group(1) or m[-1].group(2)).upper() if m else None
MODELS=['gemma2b','llama','phi','qwen','dsqwen7','qwen27','gemma31','llama70','dsllama70']
GAMES=['stag_hunt','chicken','bos','pennies']
rows=[]
for model in MODELS:
    for env in GAMES:
        for f in sorted(glob.glob(f'{ROOT}/P3_postcot/responses/{model}_{env}_seed*.json')):
            seed=int(re.search(r'seed(\d+)',f).group(1))
            d=json.load(open(f)); s=d['session']
            for r in d['rounds']:
                o=r['oracle_p_act1']
                if o is None or abs(o-0.5)<=0.10: continue
                L=parse(r.get('cot_text'))
                if L is None: continue
                if (L=='B')==(o>0.5): continue
                rows.append(dict(model=model,env=env,seed=seed,t=r['t'],oracle=o,
                    hp=s.get('hidden_param'),a0=s.get('act0'),a1=s.get('act1'),
                    letter=L,margin=abs(o-0.5),cot=r['cot_text']))
json.dump(rows,open('/tmp/claude-2770/-users-joeunyook/04be0756-aaa6-4e4b-af98-e5ce44ec8e84/scratchpad/all_fail.json','w'))
c=Counter((r['model'],r['env']) for r in rows)
print('total',len(rows))
print('model'.ljust(11)+''.join(e.rjust(11) for e in GAMES)+'    tot')
for m in MODELS:
    n=[c[(m,e)] for e in GAMES]
    print(m.ljust(11)+''.join(str(x).rjust(11) for x in n)+str(sum(n)).rjust(7))
