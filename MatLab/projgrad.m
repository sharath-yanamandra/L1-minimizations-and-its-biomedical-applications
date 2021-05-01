function [x,data] = projgrad(objfun,A,b,x0)

MAXITER   = 100;
tol       = 1e-2;
alpha_tol = 1e-6;
c         = 0.5;

data    = [];
data.x0 = x0;
data.x  = [];
data.d  = []; 
data.g  = []; 

xk = pg(x0,A,b);
nx = length(x0);

printIter();

tic
for k = 1:MAXITER
    
    xk = convert(A,xk,b);
    
    % compute function and gradient
    [f_k,g_k] = objfun(xk);

    
    % compute search direction (projected steepest descent)
    d_k = pg(xk-g_k,A,b)-xk;
    
    % terminate if this is an ascent direction
    if g_k'*d_k > 0
        printIter(k, f_k, norm(g_k), norm(d_k), g_k'*d_k, 1, toc);
        fprintf('\nExiting: step direction is not descent direction\n\n')
        if nargout == 2
        data.x = [data.x,xk];
        data.d = [data.d,d_k];
        data.g = [data.g,g_k];
        end
        break;
        % non descent direction
    end
    
    % step size
    alpha_k = 0.5;
  
    %linesearch by armijo backtracking
    xk1 = xk+alpha_k*d_k;
    f_k1 = objfun(xk1);
    
    counter = 0;
    while f_k1 > f_k + c*alpha_k*g_k'*d_k && alpha_k > alpha_tol
        % backstep alpha_k
        alpha_k = 0.5 * alpha_k;
        fprintf('Steps by linesearch %d', counter);
        % evaluate function at new iterate
        xk1 = xk + alpha_k * d_k;
        f_k1 = objfun(xk1);
        counter = counter + 1;
    end
    
    % record the iterates
    if nargout == 2
        data.x = [data.x,xk];
        data.d = [data.d,d_k];
        data.g = [data.g,g_k];
    end
    
    % update iterate
    xk = xk1;
    
    % print progress
    printIter(k, f_k, norm(g_k), norm(d_k), g_k'*d_k, alpha_k, toc);
    
    % stopping criteria
    if norm(d_k,2) < tol
        fprintf('\nExiting: step direction smaller than tolerance tol = % .4e\n\n',tol)
        break;
    end
    if alpha_k < alpha_tol
        fprintf('\nExiting: step size smaller than tolerance tol = % .4e\n\n',alpha_tol)
        break;
    end
    
end

x = xk;


end

function x = pg(y,A,b)
% project y onto the feasible set Ax <= b, i.e. nearest feasible point x to y
 b

n = length(y);
H = eye(n);
x0 = y;
options = optimset('Display','off');

x = quadprog(H,-y,A,b,[],[],[],[],x0,options);

end


function printIter(iter, f_k, g_k_norm, d_k_norm, D_k, alpha_k, CPUtime)
% print the iteration progress

if nargin==0
% Store output header and footer strings as persistent variables
out_line = '================================================================================';
out_data = '  k        f          ||g||        ||d||        g^T*d        alpha       CPU (s)';

% print algorithm output header
fprintf('\nBeginning projected gradient descent ...\n')
fprintf('%s\n%s\n%s\n', out_line, out_data, out_line)
return;
end

% Print iterate information
fprintf('% 4d  % .4e  % .4e  % .4e  % .4e  % .4e   % .5f\n',iter, f_k, g_k_norm, d_k_norm, D_k, alpha_k, CPUtime);



end