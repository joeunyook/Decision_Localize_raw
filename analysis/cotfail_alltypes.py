import json,re
from collections import defaultdict
rows=json.load(open('all2.json'))
ACT={'stag_hunt':['STAG','HARE'],'chicken':['YIELD','DARE'],'bos':['BACH','STRAVINSKY'],'pennies':['HEADS','TAILS']}
MATCH={'stag_hunt':True,'bos':True,'pennies':True,'chicken':False}
# ---- Type 2: named hypothesis on the wrong side of the observed frequency
RATE={'CAUTIOUS':.9,'WARY':.7,'BOLD':.3,'RECKLESS':.1,
      'ALWAYS_STAG':.9,'ALWAYS_HARE':.1,'ALWAYS_BACH':.9,'ALWAYS_STRAV':.1,
      'ALWAYS_HEADS':.9,'ALWAYS_TAILS':.1}
NAMED=re.compile(r'(?:most likely|likely to be|strongly suggests?|is probably|conclude that .{0,20}is|'
                 r'evidence .{0,25}suggests?)[^.]{0,70}?\b(CAUTIOUS|WARY|BOLD|RECKLESS|ALWAYS[_ ]STAG|'
                 r'ALWAYS[_ ]HARE|ALWAYS[_ ]BACH|ALWAYS[_ ]STRAV\w*|ALWAYS[_ ]HEADS|ALWAYS[_ ]TAILS)\b',re.I)
def type2(r):
    n=r['n_prev']
    if n<3: return None
    f=r['counts']['opp0']/n              # freq of the tallied action (act index 0)
    if abs(f-.5)<.2: return None
    m=list(NAMED.finditer(r['cot']))
    if not m: return None
    h=re.sub(r'[ ]','_',m[-1].group(1).upper())
    h={'ALWAYS_STRAVINSKY':'ALWAYS_STRAV'}.get(h,h)
    if h not in RATE: return None
    return (f>.5)!=(RATE[h]>.5)
# ---- Type 3: recency override / single-deviation hypothesis flip
RECENCY=re.compile(r'(most recent|the last (two|three|few|2|3) rounds|recent (rounds|behaviou?r|trend|data)|'
                   r'most telling|currently in a|trending|latest (round|data)|recent pattern)',re.I)
DEVFLIP=re.compile(r'(two .{0,25} in a row (?:is|are|makes)|statistically (un)?likely|statistically improbable|'
                   r'while mistakes (happen|are possible)|higher than the expected|too many .{0,18}mistakes|'
                   r'would require \d+ mistakes)',re.I)
# ---- Type 4: stated EU comparison contradicted by the choice
EU=re.compile(r'(?:EV|E|Expected (?:payoff|value|Payoff|Value))\s*[\(\[]?\s*'
              r'(STAG|HARE|BACH|STRAVINSKY|HEADS|TAILS|YIELD|DARE|A|B)\b[^=\n]{0,50}?'
              r'=\s*\$?\\?m?a?t?h?b?f?\{?\s*(\d+\.?\d*)',re.I)
def type4(r):
    vals={}
    for m in EU.finditer(r['cot']):
        k=m.group(1).upper(); v=float(m.group(2))
        k={'A':r['a0'].upper(),'B':r['a1'].upper()}.get(k,k)
        vals.setdefault(k,v)
    acts=[a.upper() for a in ACT[r['env']]]
    got={k:v for k,v in vals.items() if k in acts}
    if len(got)!=2: return None
    best=max(got,key=got.get)
    if abs(got[acts[0]]-got[acts[1]])<1e-9: return None
    chose=(r['a1'] if r['letter']=='B' else r['a0']).upper()
    return chose!=best
res=defaultdict(lambda: defaultdict(lambda:[0,0]))
for r in rows:
    b='wrong' if r['wrong'] else 'correct'
    t2=type2(r)
    if t2 is not None: res['T2'][b][0]+=1; res['T2'][b][1]+=t2
    for nm,rx in [('T3rec',RECENCY),('T3dev',DEVFLIP)]:
        res[nm][b][0]+=1; res[nm][b][1]+= bool(rx.search(r['cot']))
    t4=type4(r)
    if t4 is not None: res['T4'][b][0]+=1; res['T4'][b][1]+=t4
LBL={'T2':'Type2  named hypothesis on wrong side of observed freq',
     'T3rec':'Type3a recency language present',
     'T3dev':'Type3b single-deviation rejection language',
     'T4':'Type4  choice contradicts its own stated EU comparison'}
print(f'{"detector":56}{"FAILED":>18}{"CORRECT":>18}')
for k in ['T2','T3rec','T3dev','T4']:
    w=res[k]['wrong']; c=res[k]['correct']
    print(f'{LBL[k]:56}{w[1]:5}/{w[0]:<5}{100*w[1]/max(w[0],1):5.1f}%  {c[1]:5}/{c[0]:<5}{100*c[1]/max(c[0],1):5.1f}%')
