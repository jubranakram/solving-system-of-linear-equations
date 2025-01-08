import numpy as np 
import matplotlib.pyplot as plt 

from utils import gauss_seidel

if __name__ == '__main__':
    
    # Solve    
    # 2x1 -x2 = 1
    #  -x1 + 2x2 - x3 = 0
    # -x2 + x3 = 0
    
    A = np.array([[2., -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    
    b = np.array([1., 0, 0])
    # we'll call the parameter vector as s,
    # just to differentiate from the x in the quadratic equation
    x0 = np.array([0, -1, 2])
    vals = gauss_seidel(A, b, x0)
    # if matrix is singular, vals will be a string 
    if isinstance(vals, str):
        print(vals)
    else:
        sols, iters = vals
        s = sols[-1]
        print(f'The solution is: {s}')
        
        # visualization
        x = np.linspace(-3, 3, 100)
        y = s[0]*x**2 + s[1]*x + s[2]
        
        # fig, axs = plt.subplots(1, 1)
        # axs.plot(x, y, 'b', linewidth=2, alpha=0.5, label='Solution')
        # axs.plot(-2, 20, 'ok', markerfacecolor='r', alpha=0.5, label='P1')
        # axs.plot(1, 5, 'ok', markerfacecolor='g', alpha=0.5, label='P2')
        # axs.plot(3, 25, 'ok', markerfacecolor='m', alpha=0.5, label='P3')
        # axs.grid('on')
        # axs.set_xlabel('$x$')
        # axs.set_ylabel('$y$')
        # # axs.set_aspect('equal')
        # axs.legend()
        # plt.savefig('./figs/gauss_jordan_elimination_curve_fitting.jpg')