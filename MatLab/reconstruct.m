function [x_res,y_res] = reconstruct(x, n_points, interval)
     
    x_res = linspace(interval(1),interval(end),n_points);
    y_res = [];
    c = 1;
    for i = x_res
            
            y = 0;
            k = 1;
            for j = x
                y = y + j*cos(k*i);
                k = k +1;
            end
            y_res = [y_res,y];
            c = c+1;
    end
end
