from scipy.optimize import minimize
from helpers import *

A = 10.0
n = 2

f = lambda x : A * n + sum(x**2) - A * sum(np.cos(2 * np.pi * x))
g = lambda x : 2 * x + A / (2 * np.pi) * np.sin(2 * np.pi * x)
H = lambda x : 2 * np.eye(x.shape[0]) + A / (4*np.pi**2) * np.diag(np.cos(2*np.pi*x)) 

x0 = get_arg('x_init', np.array([0.1, 0.1]), lambda x : np.array(eval(x)))
d = len(x0)

tol = get_arg('tol', 1e-7, float)
max_iter = get_arg('max_iter', 1000, int)
method = get_arg('method', 'Newton-CG', str)
output_file = get_arg('output_file', 'output.pdf', str)

print('max iter = %d'%max_iter)

callback = lambda x : print(x)

#if( 'Newton' in method ):
if( True ):
    delim=' --- '
    max_hess = 5
    history = dict()
    callback = hessian_callback(f,g,H,history,delim,max_hess)


d = minimize(fun=f, method=method,
    jac=g, hess=H, x0=x0, 
    tol=tol,
    options={'maxiter': max_iter, 'disp': False},
    callback=callback
    )

print(history['g_norm'])

x_box = np.linspace(-10,10,1000)
y_box = np.linspace(-10,10,1000)
num_levels = 50

try:
    plot_progress(f, history['x'], output_file, x_box, y_box, num_levels)
except Exception as e:
    print(e)


