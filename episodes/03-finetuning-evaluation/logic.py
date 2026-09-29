import hashlib
import math
from pathlib import Path
from common import save


def best_epoch(history):
    assert history
    # Strict > retains the earlier epoch when both metrics tie.
    best=history[0]
    for row in history[1:]:
        if (row['validation']['accuracy'],row['validation']['macro_f1']) > (best['validation']['accuracy'],best['validation']['macro_f1']):
            best=row
    return best


def metrics(records, names):
    n=len(names); cm=[[0]*n for _ in range(n)]; loss=0.
    for row in records:
        probabilities=row['probs']; assert len(probabilities)==n
        assert all(math.isfinite(p) and 0<=p<=1 for p in probabilities)
        assert abs(sum(probabilities)-1)<1e-5
        predicted=max(range(n),key=probabilities.__getitem__)
        assert predicted==row['pred']; cm[row['gold']][predicted]+=1
        loss-=math.log(max(1e-12,probabilities[row['gold']]))
    per_class={}
    for i,name in enumerate(names):
        tp=cm[i][i]; support=sum(cm[i]); predicted=sum(row[i] for row in cm)
        per_class[name]={'support':support,'recall':tp/support if support else 0.,
            'precision':tp/predicted if predicted else 0.,'f1':2*tp/(support+predicted) if support+predicted else 0.}
    correct=sum(cm[i][i] for i in range(n))
    return {'n':len(records),'correct':correct,'accuracy':correct/len(records),
        'macro_f1':sum(v['f1'] for v in per_class.values())/n,'nll':loss/len(records),
        'per_class':per_class,'confusion_matrix':cm,'label_order':names}


def groups(iterator, size):
    bucket=[]
    for value in iterator:
        bucket.append(value)
        if len(bucket)==size: yield bucket; bucket=[]
    if bucket: yield bucket


def split_file(source, destination, prefix, chunk_bytes=16*1024**2):
    parts=[]; whole=hashlib.sha256(); total=0
    with Path(source).open('rb') as stream:
        while True:
            data=stream.read(chunk_bytes)
            if not data: break
            name=f'{prefix}-{len(parts):05d}.bin'; path=Path(destination)/name
            with path.open('xb') as output: output.write(data)
            digest=hashlib.sha256(data).hexdigest(); whole.update(data); total+=len(data)
            parts.append({'name':name,'bytes':len(data),'sha256':digest})
    return {'bytes':total,'sha256':whole.hexdigest(),'parts':parts}
