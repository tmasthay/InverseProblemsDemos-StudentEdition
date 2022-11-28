import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def hessian_callback(f,g,H,store,delim=' ',max_hess=5):
    keys = ['x', 'val', 'g', 'H', 'g_norm', 'U',
        'S', 'V']
    for k in keys:
        if( k not in store.keys() ):
            store[k] = []
    def helper(x):
        a = f(x)
        b = g(x)
        c = H(x)
        U,S,V = np.linalg.svd(c)
        store['x'].append(x)
        store['val'].append(a)
        store['g'].append(b)
        store['H'].append(c)
        store['g_norm'].append(np.linalg.norm(b))
        store['U'].append(U)
        store['S'].append(S)
        store['V'].append(V)
        if( len(S) > max_hess ):
            S = S[:max_hess]
        print('x=%s%sf=%.3e%sg_norm=%.3e%ssigma_h=%s\n'%(
            str(x), delim,
            a, delim,
            np.linalg.norm(b), delim,
            '[' + ','.join(['%.3e'%e for e in S]) + ']'))
    return helper
        
def plot_progress(f, x_history, output_name, x_box, y_box, num_levels):
    if( len(x_history) == 0 ):
        raise(Exception('Empty history list...cannot plot'))

    if( len(x_history[0]) > 2 ):
        raise(Exception('Domain of dimension %d > 2...cannot plot'%(len(x_history[0]))))

    if( len(x_history[0]) == 1 ):
        x_history = np.ndarray.flatten(np.array(x_history))
        x_min = min(x_history)
        x_max = max(x_history)
        x = np.array([np.array([e]) for e in np.linspace(x_min, x_max, N)])
        colors = cm.rainbow(np.linspace(0,1,len(x_history)))
        for xx, c in zip(x_history, colors):
            plt.scatter( xx, f(np.array([xx])), color=c )
        plt.plot(np.ndarray.flatten(x),np.array(list(map(f,x))))
        plt.xlabel('X')
        plt.ylabel('f')
        plt.savefig('%s.pdf'%output_name.replace('.pdf',''))
    else:
        x_coords = [e[0] for e in x_history]
        y_coords = [e[1] for e in x_history]

        X,Y = np.meshgrid(x_box,y_box) 

        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i][j] = f(np.array([X[i][j], Y[i][j]]))

        bar = plt.contourf(X,Y,Z, num_levels)
        plt.colorbar(bar)

        colors = cm.rainbow(np.linspace(0,1,len(x_history)))
        for xx, c in zip(x_history, colors):
            plt.scatter( xx[0], xx[1], color=c )
        plt.title(output_name.replace('.pdf',''))
        plt.savefig('%s.pdf'%(output_name.replace('.pdf','')))         
