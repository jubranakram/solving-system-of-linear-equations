import numpy as np 
from utils import check_pivots, gauss_elimination


if __name__ == '__main__':
    # A = np.array([[0., -1, 1],
    #               [-1, 2, -1],                  
    #               [2, -1, 0]])
    
    # b = np.array([0, 0, 1.])
    
    # A = np.array([[4, -2, 1.],
    #               [-2, 4, -2],                  
    #               [1, -2, 4]])
    
    # b = np.array([11., -16, 17])
    
    A = np.array([[4., -2, 1],
                  [1, 1, 1],
                  [9, 3, 1]])
    
    b = np.array([20., 5, 25])
    
    print("Original A:")
    print(A)
    print("Original b")
    print(b)
    
    A, b, _ = check_pivots(A, b)
    print("After pivoting A:")
    print(A)
    print("After pivoting b")
    print(b)
    
    A, b, x = gauss_elimination(A, b)
    print("U:")
    print(A)
    print("c")
    print(b)
    print("The solution is: ")
    print(x)