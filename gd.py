import numpy as np
from helpers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from typing import Callable

def gd(**kw):
    #partially evaluate default parameters
    kwd = kw_defaults(kw)

    #REQUIRED arguments
    f = kw['f']
    g = kw['grad']
    x0 = kw['x0']

    #OPTIONAL arguments
    tau = kwd('tau', 1e-15)
    c_armijo = kwd('c_armijo', 1e-4)
    c_curvature = kwd('c_curvature', 1e-2)
    alpha = kwd('alpha', 1.0)
    mode = kwd('mode', 'backtracking')
    max_iter = kwd('max_iter', 100)
    verbose = kwd('verbose', True)
    stochastic = kwd('stochastic', lambda x,it: np.zeros(x0.shape))

    #initialize current step and step length
    d = len(x0)
    x = x0
    beta = alpha

    #initialize return values
    x_history = [x]
    f_history = [f(x)]
    r_history = []

    #main loop
    it = 0
    while( it < max_iter ):
        #Create verbose print statements
        if( verbose ):
            if( it > 0 ):
                print('x: %s     f: %f      grad: %s    r: %f    step_length: %f'%(x, f_history[-1], g(x), r_history[-1], beta))
            else:
                print('x: %s    f: %f    grad: %s    start_beta: %f'%(x, f_history[-1], g(x), beta))

        #define perturbation phi and associated Wolfe condition functions
        phi = lambda gamma : f(x - gamma * g(x))
        phi_prime = lambda gamma : #TODO: implement derivative of phi
        armijo = lambda gamma : #TODO: implement armijo condition
        curv = #TODO: implement curvature condition

        #make loop variable beta so we don't accidentally overwrite alpha
        beta = alpha

        if( mode == 'backtracking' ):
            #TODO: what condition must be violated 
            #    for us to backtrack?
            while(  ):
                #TODO: implement backtracking step 
        elif( mode == 'exact' ):
            #NOTE: exact line search is implemented for you here
            #    notice that we called the outside package scipy
            #    to solve an AUXILIARY OPTIMIZATION PROBLEM!
            tmp = minimize(phi, 0.0)
            beta = tmp['x'][0]
            
        #TODO: add the stochastic term to implement SGD
        #    NOTE that the default of zeros for the stochastic term 
        #        recovers "vanilla gradient descent."
        x = x - beta * g(x)

        #store info
        x_history.append(x)
        f_history.append(f(x))
        r_history.append(abs(f_history[-2] - f_history[-1]))

        it += 1
   
        #if our progress is smaller than cutoff, terminate
        if( r_history[-1] < tau ):
            break

    return {'x' : x_history[-1], 
        'val' : f_history[-1],
        'x_history': x_history,
        'f_history': f_history,
        'residual' : r_history,
        'iterations': it}

#function for plotting what our optimization looked like
#     f: the function we optimized
#     x_history: sequence of guesses in optimization
#     output_name: the file to store the plot
#     x_box, y_box: define domain to draw contours of f
#     num_levels: number of contours we want on plot
def plot_progress(f, x_history, output_name, x_box, y_box, num_levels):
    #sanity check x_history input
    if( len(x_history) == 0 ):
        raise(Exception('Empty history list...cannot plot'))

    #Only supports plotting in 1D and 2D
    if( len(x_history[0]) > 2 ):
        raise(Exception('Domain of dimension %d > 2...cannot plot'%(len(x_history[0]))))

    #Handle 1D case
    if( len(x_history[0]) == 1 ):
        #flatten 1D points
        x_history = np.ndarray.flatten(np.array(x_history))

        #define interval to move through
        x_min = min(x_history)
        x_max = max(x_history)
        x = np.array([np.array([e]) for e in np.linspace(x_min, x_max, N)])

        #draw sequence of guesses
        #    First guess: in violet
        #    Last guess: in red
        #    Intermediate guesses: interpolated through ends of rainbow
        #        traversing from violet to red
        colors = cm.rainbow(np.linspace(0,1,len(x_history)))
        for xx, c in zip(x_history, colors):
            plt.scatter( xx, f(np.array([xx])), color=c )

        #plot the optimization landscape
        plt.plot(np.ndarray.flatten(x),np.array(list(map(f,x))))
        plt.xlabel('X')
        plt.ylabel('f')

        #save figure
        plt.savefig('%s.pdf'%output_name.replace('.pdf',''))
    #handle 2D case
    else:
        #grab coordinates and make grid
        x_coords = [e[0] for e in x_history]
        y_coords = [e[1] for e in x_history]
        X,Y = np.meshgrid(x_box,y_box) 

        #evaluate function on grid
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i][j] = f(np.array([X[i][j], Y[i][j]]))

        #make the contour plot and color bar
        bar = plt.contourf(X,Y,Z, num_levels)
        plt.colorbar(bar)

        #draw sequence of guesses
        #    First guess: in violet
        #    Last guess: in red
        #    Intermediate guesses: interpolated through ends of rainbow
        #        traversing from violet to red
        colors = cm.rainbow(np.linspace(0,1,len(x_history)))
        for xx, c in zip(x_history, colors):
            plt.scatter( xx[0], xx[1], color=c )

        #save figure
        plt.savefig('%s.pdf'%(output_name.replace('.pdf','')))
