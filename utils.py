import numpy as np 
import matplotlib.pyplot as plt 

def plot_slope_intercept_line(x, m, c, 
                              axs=None, color='g', linewidth=2, 
                              linestyle='solid', alpha=0.5):
    
    # calculate the y values 
    y = m*x + c
    
    # plot line    
    if axs is None:
        fig, axs = plt.subplots(1, 1)
    axs.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    return axs

def check_pivots(A, b):
    '''
    Updates the coefficient matrix A for diagonal dominance
    '''
    
    # number of rows and columns (m, n)
    m, n = A.shape   
    
    # scale factor for each row
    s = np.max(np.abs(A), axis=1)
    # pivoting loop
    for idx in range(n-1):
        # check the relative size of elements in the each column
        # and swap rows if there's an element that has the largest
        # relative size than the diagonal element at (idx, idx)
        nidx = np.argmax(np.abs(A[idx:, idx])/s[idx])
        if nidx:
            nidx = nidx + idx 
            # rows swap
            current_A = A[idx].copy()
            current_b = b[idx].copy()
            current_s = s[idx].copy()
            A[idx], A[nidx] = A[nidx], current_A
            b[idx], b[nidx] = b[nidx], current_b
            s[idx], s[nidx] = s[nidx], current_s
            
        if A[idx, idx] < np.finfo(np.float64).eps:
            return A, b, True
            
    return A, b, False

def forward_elimination(A, b):
    
    # number of rows and columns (m, n)
    m, n = A.shape
    
    # Pivoting
    
    # scale factor for each row
    s = np.max(np.abs(A), axis=1)
    # pivoting loop
    for idx in range(n-1):
        # check the relative size of elements in the each column
        # and swap rows if there's an element that has the largest
        # relative size than the diagonal element at (idx, idx)
        nidx = np.argmax(np.abs(A[idx:, idx])/s[idx])
        if nidx:
            nidx = nidx + idx 
            # rows swap
            current_A = A[idx].copy()
            current_b = b[idx].copy()
            current_s = s[idx].copy()
            A[idx], A[nidx] = A[nidx], current_A
            b[idx], b[nidx] = b[nidx], current_b
            s[idx], s[nidx] = s[nidx], current_s
            
        if np.abs(A[idx, idx])/s[idx] < np.finfo(np.float64).eps:
            return "Singular matrix and no unique solution"
        
        # Elimination phase
        lambdas = A[idx+1:, idx]/A[idx, idx]
        A[idx+1:, idx:] = A[idx+1:, idx:] - np.outer(lambdas, A[idx, idx:])
        b[idx+1:] = b[idx+1:] - lambdas*b[idx]
        
    return A, b

def backward_elimination(A, b):
    
    # number of rows and columns (m, n)
    m, n = A.shape
    
    for idx in range(n-1, -1, -1):
        # convert diagonal element to 1
        b[idx] = b[idx]/A[idx, idx]
        A[idx, idx] = 1.
        # eliminate all elements in the column above the diagonal element
        lambdas = A[:idx, idx].copy()
        A[:idx, idx] = 0
        b[:idx] = b[:idx] - lambdas*b[idx]
        
    return A, b
    
    
    

def gauss_elimination(A, b):
    '''
    Solves a system of linear equations
    Ax = b
    using Gauss elimination method
    
    Inputs:
    
    A: Coefficient matrix
    b: Output vector
    '''
    
    # Forward elimination
    vals = forward_elimination(A, b)
    
    if isinstance(vals, str):
        return vals
    
    A, b = vals
    
    # number of rows and columns or equations and unknowns
    m, n = A.shape
        
    x = np.zeros_like(b)
    # Back substitution phase
    for idx in range(n-1, -1, -1):
        x[idx] = (b[idx] - np.dot(A[idx,idx+1:], x[idx+1:]))/A[idx, idx]
    
    return A, b, x

def gauss_jordan_elimination(A, b):
    '''
    Solves a system of linear equations
    Ax = b
    using Gauss-Jordan elimination method
    
    Inputs:
    
    A: Coefficient matrix
    b: Output vector
    '''
    
    # Forward elimination
    vals = forward_elimination(A, b)
    
    if isinstance(vals, str):
        return vals
    
    A, b = vals
    
    # Backward elimination
    A, b = backward_elimination(A, b)
    
    x = b.copy()
    
    return A, b, x

def forward_substitution_unitdiag(A, b):
    # number of rows and columns (equations and unknowns)
    m, _ = A.shape
    y = b.copy()
    for k in range(1,m):
        y[k] = b[k] - np.sum(A[k,:k]*y[:k])
    return y

def backward_substitution_nonunitdiag(A, b):
    _, n = A.shape
    x = np.zeros_like(b)
    # Back substitution phase
    for idx in range(n-1, -1, -1):
        x[idx] = (b[idx] - np.dot(A[idx,idx+1:], x[idx+1:]))/A[idx, idx]
    return x
        
    

def doolittle_decomposition(A, b):
    '''
    Computes LU decomposition of a matrix A
    '''
    
    # number of rows and columns (equations and unknowns)
    m, n = A.shape
    # first row of A = first row of U
    for ridx in range(1, m):                
        for cidx in range(n):   
            # if i > j, update elements of L         
            if (ridx > cidx):                
                A[ridx, cidx] = (A[ridx, cidx] - np.sum(A[ridx, :cidx]*A[:cidx, cidx]))/A[cidx, cidx]
            else: # if i <= j, update elements of U
                A[ridx, cidx] = A[ridx, cidx] - np.sum(A[ridx, :ridx]*A[:ridx, cidx])
    y = forward_substitution_unitdiag(A, b)
    x = backward_substitution_nonunitdiag(A, y)
    
    return A, y, x

def forward_substitution_nonunitdiag(A, b):
    # number of rows and columns (equations and unknowns)
    m, _ = A.shape
    y = b.copy()
    for k in range(m):
        y[k] = (b[k] - np.sum(A[k,:k]*y[:k]))/A[k, k]
    return y

def backward_substitution_unitdiag(A, b):
    _, n = A.shape
    x = np.zeros_like(b)
    # Back substitution phase
    for idx in range(n-1, -1, -1):
        x[idx] = b[idx] - np.dot(A[idx,idx+1:], x[idx+1:])
    return x

def crout_decomposition(A, b):
    '''
    Computes LU decomposition of a matrix A
    '''
    
    # number of rows and columns (equations and unknowns)
    m, n = A.shape    
    for ridx in range(m):                
        for cidx in range(n):   
            # if i >= j, update elements of L         
            if (ridx >= cidx):                
                A[ridx, cidx] = A[ridx, cidx] - np.sum(A[ridx, :cidx]*A[:cidx, cidx])
            else: # if i < j, update elements of U
                A[ridx, cidx] = (A[ridx, cidx] - np.sum(A[ridx, :ridx]*A[:ridx, cidx]))/A[ridx, ridx]
    y = forward_substitution_nonunitdiag(A, b)
    x = backward_substitution_unitdiag(A, y)
    
    return A, y, x

def is_symmetric(A):
    '''
    Checks if A is symmetric, which is true when A = transpose(A)
    '''
    return np.allclose(A, A.T)

def is_positive_definite(A):
    '''
    Checks if A is positive definite, which is true when principal minors are all positive
    '''
    _, n = A.shape
    for idx in range(1, n+1):
        pminor = A[:idx, :idx]
        # check if determinant is negative or zero, if True, then return False
        if np.linalg.det(pminor) <= 0:
            return False
    return True

def cholesky_decomposition(A, b):
    '''
    Computes LU decomposition of a matrix A
    '''   
    # number of rows and columns (equations and unknowns)
    m, n = A.shape
    # check the first requirement: A is symmetric
    if not is_symmetric(A):
        return "A is not a symmetric matrix"
    # check the second requirement: A is positive definite
    if not is_positive_definite(A):
        return "A is not positive definite"
    
    M = A.copy()
    
    for ridx in range(m):
        for cidx in range(ridx+1):
            # diagonal elements if ridx = cidx
            if ridx == cidx:
                M[ridx, cidx] = np.sqrt(A[ridx, cidx] - np.sum(M[ridx, :cidx]**2))
            else:
                M[ridx, cidx] = (A[ridx, cidx] - np.sum(M[ridx, :cidx]*M[cidx, :cidx]))/M[cidx, cidx]
                M[cidx, ridx] = M[ridx, cidx].copy()
                
    y = forward_substitution_nonunitdiag(M, b)
    x = backward_substitution_nonunitdiag(M, y)
    
    return M, y, x

def gauss_seidel(A, b, x0, max_iters=100, epsilon=1e-6):
    '''
    Finds the solution of a systems of linear equation
    using Gauss-Seidel's method (an iterative scheme)
    '''   
    # number of rows and columns (equations and unknowns)
    m, _ = A.shape
    # Solutions
    sols = [x0.copy(),]
    iters = 0
    while iters < max_iters:
        for idx in range(m):
            # update
            x0[idx] = (b[idx] - np.dot(A[idx, :idx], x0[:idx])- np.dot(A[idx, idx+1:], x0[idx+1:])) / A[idx, idx]
        sols.append(x0.copy())
        iters += 1
        if np.linalg.norm(x0 - sols[iters-1], np.inf) < epsilon:            
            break
    return sols, iters

def conjugate_gradient(A, b, x0, max_iters=100, epsilon=1e-6):
    '''
    Finds the solution of a systems of linear equation
    using the Conjugate Gradient method (an iterative scheme)
    '''   
    # number of rows and columns (equations and unknowns)
    m, _ = A.shape
    # Solutions
    sols = [x0.copy(),]
    
    if not (is_symmetric(A) and is_positive_definite(A)):
        print("A is not symmetric positive definite - Conjugate gradient might work but convergence is not guaranteed")
        
    # Initialization
    r0 = b - np.dot(A, x0)
    p0 = r0.copy()
    
    iters = 0
    while iters < max_iters:
        # step size
        r0_olddot = np.dot(r0.T, r0)
        alpha = r0_olddot / np.dot(p0.T, A.dot(p0))
        # update solution
        x0 = sols[iters] + alpha*p0
        sols.append(x0.copy())
        # update residual
        r0 = b - np.dot(A, x0)
        iters += 1
        # check stopping criterion
        if np.linalg.norm(r0, np.inf) < epsilon:
            break
        # conjugate direction coefficient
        beta = np.dot(r0.T, r0) / r0_olddot
        # conjugate search direction
        p0 = r0 + beta*p0 
        
        
    return sols, iters

def qr_decomposition(A: np.ndarray, b: np.ndarray):
    '''
    Solves a system of linear equations using QR decomposition of a matrix A
    '''

    # Step 1: QR decomposition
    Q, R = np.linalg.qr(A)

    # Step 2: Solve R * x = Q^T * b = y using backward substitution
    y = np.dot(Q.T, b)
    x = backward_substitution_nonunitdiag(R, y)  

    return Q, R, x
        
        
        
            
    
    
    
       
    
    
    
    