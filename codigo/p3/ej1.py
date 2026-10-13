
def solve_eliptic(p, r, f, a, b, c, d, n, m, eq_solver):
    import pdb
    pdb.set_trace()
    U = np.zeros((m+1, n+1))
    hx = (b-a)/n
    hy = (d-c)/m
    x = np.linspace(a,b,n+1)
    y = np.linspace(c,d,m+1)
    X, Y = np.meshgrid(x, y)
    # diagonal k=1
    A1 = sps.diags_array(np.ones(n-1)*(1/(hx*hx)),offsets=1)
    # diagonal k=-1
    A2_diag = np.ones(n-1)
    A2_diag[-1] = 2
    A2_diag *= 1/(hx*hx)
    A2 = sps.diags_array(A2_diag,offsets=-1)
    # ambas diagonales, k=1, -1
    A_1_2 = sps.block_diag([A1+A2 for _ in range(m-1)])
    # diagonal principal
    R = np.vectorize(r)
    P = np.vectorize(p)
    A3_diag_mat = R(X,Y) + P(X,Y)*(-2/(hy*hy)) - 2/(hx*hx)
    A3_diag_mat = A3_diag_mat[1:-1,1:]
    A3_diag = A3_diag_mat.flatten()
    A3 = sps.diags_array(A3_diag)
    # diags k = -1, 0, 1
    A_1_2_3 = A_1_2 + A3
    # diagas k = -n, n
    P_diag_mat = P(X,Y)/(hy*hy)
    P_diag_mat = P_diag_mat[1:-1,1:]
    P_diag = P_diag_mat.flatten()
    P_diag_up = P_diag[:-n]
    P_diag_down = P_diag[n:]
    P_sparse = sps.diags_array([P_diag_down, P_diag_up], offsets=[-n,n])
    # final ensamble
    A = P_sparse + A_1_2_3
    # vector F
    F = np.vectorize(f)
    F_mat = F(X,Y)
    F_mat[:,-1] = F_mat[:,-1] - 2*(-np.pi*np.sin(np.pi*y))/hx
    F_mat = F_mat[1:-1,1:]
    F_vec = F_mat.flatten()
    U[1:-1,1:] = eq_solver(A, F_vec).reshape(m-1,n)
    return U, X, Y
