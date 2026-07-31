import json,re
from collections import defaultdict,Counter
rows=json.load(open('all2.json'))
ACT={'stag_hunt':['STAG','HARE'],'chicken':['YIELD','DARE'],'bos':['BACH','STRAVINSKY'],'pennies':['HEADS','TAILS']}
TALLIED={'stag_hunt':'STAG','chicken':'YIELD','bos':'BACH','pennies':'HEADS'}
NUM=re.compile(r'(\d{1,2})\s*(?:out of|/)\s*(?:the\s+)?(\d{1,2})')
SUBJVERB=re.compile(r'(?:opponent|colleague|hunter|they|the other)[^.]{0,70}?'
                    r'(?:chose|played|hunted|yielded|dared|has chosen|have chosen|has played|'
                    r'have played|has yielded|selected|picked)',re.I)
HYPO=re.compile(r'\b(would|if |were |expect|strategy would|assum|suppose)',re.I)
def claims(txt,env,n):
    acts=ACT[env]; found={}
    for m in NUM.finditer(txt):
        x,y=int(m.group(1)),int(m.group(2))
        if y!=n or x>y: continue
        win=txt[max(0,m.start()-30):m.end()+18]
        hits=[a for a in acts if re.search(rf'\b{a}\b',win,re.I)]
        if len(hits)!=1: continue                      # ambiguous -> skip
        pre=txt[max(0,m.start()-160):m.start()]
        if not SUBJVERB.search(pre): continue          # must assert opponent behaviour
        if HYPO.search(txt[max(0,m.start()-110):m.end()+18]): continue
        found.setdefault(hits[0],(x,y))                # first claim per action only
    return found
per=defaultdict(lambda:[0,0]); kind=Counter(); ex=[]
for r in rows:
    if not r['wrong']: continue
    n=r['n_prev']; c=r['counts']; a0,a1=ACT[r['env']]
    true={a0:c['opp0'],a1:c['opp1']}; mine={a0:c['my0'],a1:c['my1']}; other={a0:a1,a1:a0}
    for a,(x,y) in claims(r['cot'],r['env'],n).items():
        per[r['model']][0]+=1
        if x==true[a]: continue
        per[r['model']][1]+=1
        if x==true[other[a]]: k='swapped with other action'
        elif x==mine[a]:      k='reported own count'
        else:                 k='free fabrication'
        kind[k]+=1
        if len(ex)<200: ex.append((r['model'],r['env'],r['seed'],r['t'],a,x,y,true[a],k,
            r['cot'][max(0,r['cot'].lower().find(a.lower())-120):][:250].replace('\n',' ')))
tot=[0,0]
print('TALLY CONTRADICTION RATE on failed rounds (dedup, nearest-action, hypotheticals excluded)')
print('  the prompt PRINTS this count: "# Tally -> opponent played %s N/M rounds"'%'<act>')
print('  model        claims  contradict   rate')
for m in ['gemma2b','llama','phi','qwen','dsqwen7','qwen27','gemma31','llama70','dsllama70']:
    a,b=per[m]; tot[0]+=a; tot[1]+=b
    if a>=5: print(f'  {m:11}{a:7}{b:9}{100*b/a:8.1f}%')
print(f'  {"TOTAL":11}{tot[0]:7}{tot[1]:9}{100*tot[1]/tot[0]:8.1f}%')
print('\n  kind:', dict(kind))
json.dump(ex,open('miscount3.json','w'))
