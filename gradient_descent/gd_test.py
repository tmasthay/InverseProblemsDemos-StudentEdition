from gd import *

f = lambda x : #TODO: implement 2D Rastrigin function
g = lambda x : #TODO: implement gradient

x0 = np.array([2.0, 3.0])
tau = 1e-10
c_armijo = #TODO: choose armijo parameter
c_curvature = #TODO: choose curvature condition parameter
max_iter = 1000
mode = 'backtracking'

sig_max = 0.0 #TODO: set to nonzero value to implement SGD
sig_min = 0.0 #TODO: set to value less than or equal to sig_max to implement SGD

stochastic = lambda x,it : np.random.randn(2) * ( (max_iter-it) * sig_max + it * sig_min )

d = gd(f=f, grad=g, x0=x0, 
    tau=tau, 
    c_armijo=c_armijo, 
    c_curvature=c_curvature, 
    max_iter=max_iter,
    mode=mode,
    stochastic=stochastic)

print('Approx min of %f at %s after %d iterations'%(d['val'], d['x'], d['iterations']))

x_box = np.linspace(-2,2,100)
y_box = np.linspace(-2,2,100)
num_levels = 100
plot_progress(f, d['x_history'], 'quadratic', x_box, y_box, num_levels)


