function [x,y] = dualsimplex(A,b,c,B,tol_set, tol_opt, MAX_ITER,state,verbose)

[m, n] = size(A);
ROWS = linspace(1,m,m);

iteration  = 1; ref = 2;

%Default basis if it is not provided
if ~length(B)  
    B = ROWS(1:n);
end

N = setdiff(ROWS, B);

while(state == "" & MAX_ITER > iteration)
    
    A_b = A(B,:); 
    A_N = A(N,:);
    b_b = b(B); 
    b_N = b(N);
    y = zeros(1,m);
    
    %Initialization of x_ and y_
    
    %======Normal solving======%
    %x =  A_b\b_b;
    
    %======LU Solving======%
    if(iteration == 1 || mod(iteration, ref) == 0)
        [L_r,U_r] = lu(A_b');
    else
        [L_r,U_r] = forrest_tomlin(A(k,:)', q, L_r, U_r);
        [L,U] = deal(U_r',L_r');
    end
    
    [L,U] = deal(U_r', L_r');

    y_LU = L\b_b;
    x = U\y_LU;

    %======Normal solving======%
    %y(B) = c/A_b;

    %======LU Solving======%
    lambda_LU = U'\c';
    z_LU = L'\lambda_LU;
    y(B) = z_LU';

    %Checking optimal condition
    A_nx = A_N * x;

    if all(A_nx + tol_opt <= b_N)
        fprintf('||--------Optimal solution found--------||\n');
        state = "optimal";
        break
    end

    %Finding the entering variable
    min_k = find(A_nx > b_N, 1, 'first');

    k = N(min_k);

    eta_b = A(k,:)/A_b;

    %Primal empty check
    if all(eta_b <= tol_set)
        state = "P.empty";
        fprintf('||--------Primal set is empty [%s]--------||\n', state);
        break
    end

    %Finding the exiting variable
    theta = y(B)./eta_b;

    [v, i] = min(theta(eta_b>0));
    i = find(theta == v & eta_b > 0,1);
    h = B(i);

    %[DEBUG] Printing variables as in the order of Exercise 2.31
    if verbose
        B
        x
        A_nx
        b_N
        y
        k
        eta_b
        h
    end        

    q = find(B==h);    
    B(q)=k;	
    N = setdiff(ROWS, B);
    iteration = iteration + 1; 
    
end
end
