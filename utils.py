import numpy as np

def rerange(ori,maplist):
    newstr=list('I'*27)
    for k,v in maplist.items():
        newstr[26-v]=ori[11-k]
    return ''.join(newstr)

# average slope method
def avs(results):
    n = len(results)
    results = np.array(results)
    v0 = results[0]
    if n == 1:
        return v0    
    v1 = results[1:]
    coeff = np.array([1/i for i in range(1, n)])
    value = v0 + ((v0 - v1) * coeff).mean()
    return value 