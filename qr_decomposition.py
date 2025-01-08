# QR decomposition for solving linear equations
# Author: Jubran Akram

import numpy as np
from utils import qr_decomposition

if __name__ == '__main__':
    A = np.array([[3., 4, -2],
                  [2, -3, 4],
                  [1, -2, 3]])
    
    b = np.array([0., 11, 7])
    
    Q, R, x = qr_decomposition(A, b)

    # Solution

    # Q:
    # [[-0.80178373  0.59546516  0.05063697]
    # [-0.53452248 -0.67666495 -0.50636968]
    # [-0.26726124 -0.43306557  0.86082846]]
    # R:
    # [[-3.74165739 -1.06904497 -1.33630621]
    # [ 0.          5.27798663 -5.19678683]
    # [ 0.          0.          0.45573272]]
    # x:
    # [ 2. -1.  1.]
    print(f"Is A = QR? : {'Yes' if np.allclose(np.dot(Q, R), A) else 'No'}")
    print(f"Is x the correct solution? : {'Yes' if np.allclose(np.dot(A, x), b) else 'No'}")
    print(f"Q: {Q}")
    print(f"R: {R}")
    print(f"x: {x}")
    