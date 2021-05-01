def projgrad(objfun, A,b,x0):
    """
    PGD
    min f(x)
    s.t. Ax <= b
    """

    # Parameters
    MAXITER   = 100
    tol       = 1e-2
    alpha_tol = 1e-6
    c         = 0.5

    data    = {}
    data["x0"] = x0
    data["x"]  = [] 
    data["d"]  = [] 
    data["g"]  = []

    #x_o feasibility
    xk = pg(x0,A,b)
    nx = x0.shape[0]

    for k in range(1,MAXITER+1):
        
        #Function and Gradient
        f_k,g_k = objfun(xk)

        # Search direction (projected steepest descent)
        d_k = pg(xk-g_k,A,b)-xk
        
        # Check ascent direction
        if np.dot(g_k,d_k)[0][0] > 0

            data["x"]  = [data.x,xk]
            data["d"]  = [data.d,d_k]
            data["g"]  = [data.g,g_k]

            print(k, f_k, norm(g_k), norm(d_k), g_k.T*d_k, 1)
            print('\nExiting: step direction is not descent direction\n\n')
            break
        
        alpha_k = 1

        # linesearch by armijo backtracking
        xk1 = xk+alpha_k*d_k
        f_k1 = objfun(xk1)
        while (f_k1 > f_k + c*alpha_k*g_k.T*d_k and alpha_k > alpha_tol):
            # backstep alpha_k
            alpha_k = 0.5 * alpha_k

            # evaluate function at new iterate
            xk1 = xk + alpha_k * d_k
            f_k1 = objfun(xk1)

        # record the iterates
        data["x"]  = [data.x,xk]
        data["d"]  = [data.d,d_k]
        data["g"]  = [data.g,g_k]

        xk = xk1
        
        print(k, f_k, norm(g_k), norm(d_k), g_k.T*d_k, alpha_k, toc)
        
        # Stop criteria
        if np.linalg.norm(d_k) < tol
            print('\nExiting: step direction smaller than tolerance tol = # .4e\n\n',tol)
            break
        
        if alpha_k < alpha_tol
            print('\nExiting: step size smaller than tolerance tol = # .4e\n\n',alpha_tol)
            break

    return xk

def pg(y,A,b):
# project y onto the feasible set 
# minimize 0.5||x-y||^2
# s.t. Ax <= b   


n = length(y)
H = np.identity(n)
x0 = y
options = optimset('Display','off')
x = quadprog(H,-y,A,b,[],[],[],[],x0,options)

function printIter(iter, f_k, g_k_norm, d_k_norm, D_k, alpha_k, CPUtime)

if nargin==0
out_line = '================================================================================'
out_data = '  k        f          ||g||        ||d||        g^T*d        alpha       CPU (s)'

print('\nBeginning projected gradient descent ...\n')
print('%s\n%s\n%s\n', out_line, out_data, out_line)
return

print('# 4d  # .4e  # .4e  # .4e  # .4e  # .4e   # .5f\n',iter, f_k, g_k_norm, d_k_norm, D_k, alpha_k, CPUtime)



