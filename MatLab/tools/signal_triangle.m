function y = signal_triangle(x)
A = 2; F = 1;
y = A*sawtooth(2*pi*F*x,0.5);
end
