import sys

def kw_defaults(kw):
    def helper(key, val):
        return val if key not in kw.keys() else kw[key]
    return helper 

def get_arg(key, val, f=str):
    s = ' '.join(sys.argv)
    t = '%s='%key
    if( t in s ):
        return f(s.split(t)[-1].split(' ')[0])
    else:
        return val
