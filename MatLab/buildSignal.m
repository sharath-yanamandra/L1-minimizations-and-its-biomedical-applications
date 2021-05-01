%Sine Function implementation Report (1) Problem
function [A,b,x_m,y_m] = buildSignal(m,n,sign_function,interval)  
    
    x_n = linspace(interval(1), interval(end),n)';
    disp(size(x_n));
    size(x_n)
    y_n = sign_function(x_n)';
    
    perm = randperm(length(x_n),m);
    x_m = x_n(perm);
    y_m = sign_function(x_m)';
    
    A = rand(m,n);
    b = sign_function(x_m)';

    %Initializing A matrix according to sin(j*x_i)
    for i = 1:m
        for j = 1:n
            A(i,j) = cos(j*x_m(i));
        end
    end
end