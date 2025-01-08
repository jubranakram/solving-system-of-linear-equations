import numpy as np 
import matplotlib.pyplot as plt 

from utils import doolittle_decomposition

if __name__ == '__main__':
    
    # Curve fitting
    # Find a quadratic equation y = ax**2 + bx + c
    # whose graph passes through (-2, 20), (1, 5) and (3, 25)
    
    # Step 1: system of equations
    # 4a -2b + c = 20
    #  a + b + c = 5
    # 9a + 3b + c = 25
    
    A = np.array([[4., -2, 1],
                  [1, 1, 1],
                  [9, 3, 1]])
    
    b = np.array([20., 5, 25])
    # we'll call the parameter vector as s,
    # just to differentiate from the x in the quadratic equation
    vals = doolittle_decomposition(A, b)
    # if matrix is singular, vals will be a string 
    if isinstance(vals, str):
        print(vals)
    else:
        A, b, s = vals
        print(f'The solution is: {s}')
        
        # visualization
        x = np.linspace(-3, 3, 100)
        y = s[0]*x**2 + s[1]*x + s[2]
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(x, y, 'b', linewidth=2, alpha=0.5, label='Solution')
        axs.plot(-2, 20, 'ok', markerfacecolor='r', alpha=0.5, label='P1')
        axs.plot(1, 5, 'ok', markerfacecolor='g', alpha=0.5, label='P2')
        axs.plot(3, 25, 'ok', markerfacecolor='m', alpha=0.5, label='P3')
        axs.grid('on')
        axs.set_xlabel('$x$')
        axs.set_ylabel('$y$')
        # axs.set_aspect('equal')
        axs.legend()
        plt.savefig('./figs/gauss_jordan_elimination_curve_fitting.jpg')