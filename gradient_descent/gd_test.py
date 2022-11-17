from gd import *

A = 10.0
n = 2

f = lambda x : #TODO: Implement Rastrigin function
g = lambda x : #TODO: Implement Rastrigin function gradient

#Setup initial condition
x0 = get_arg('x_init', np.array([2.0, 3.0]), lambda x : np.array(eval(x)))
d = len(x0)

#TODO: Tune parameters for better convergence! Try higher dimensions!
#tolerance level
tau = get_arg('tau', 1e-7, float)

#armijo constant
c_armijo = get_arg('c_armijo', 1e-4, float)

#curvature constant -- note that it must be greater than armijo constant!
c_curvature = get_arg('c_curvature', 1e-2, float)

#max iterations
max_iter = get_arg('max_iter', 1000, int)

#mode -- either "exact" or "backtracking"
mode = get_arg('mode', 'exact', str)

#stochastic terms
sig_max = get_arg('sig_max', 0.0, float)
sig_min = get_arg('sig_min', 0.0, float)
output_file = get_arg('output_file', 'output', str)

#TODO: modify rate of decay of stochastic term with iteration number
stochastic = lambda x,it : np.random.randn(d) * ( (max_iter-it) * sig_max + it * sig_min )

#run program and get output
d = gd(f=f, grad=g, x0=x0, 
    tau=tau, 
    c_armijo=c_armijo, 
    c_curvature=c_curvature, 
    max_iter=max_iter,
    mode=mode,
    stochastic=stochastic)

#print final results
print('Approx min of %f at %s after %d iterations'%(d['val'], d['x'], d['iterations']))

#define plotting window
x_box = np.linspace(-10,10,1000)
y_box = np.linspace(-10,10,1000)
num_levels = 50

#plot, if possible
try:
    plot_progress(f, d['x_history'], output_file, x_box, y_box, num_levels)
except Exception as e:
    print(e)


