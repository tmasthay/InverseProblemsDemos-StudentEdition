
def kw_defaults(kw):
    def helper(key, val):
        return val if key not in kw.keys() else kw[key]
    return helper
    
