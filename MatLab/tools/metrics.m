function [mse, corr]= metrics(sol, tar)

mse = mean((sol - tar).^2);
corr = corrcoef(sol, tar);
corr = corr(1,2);
end


