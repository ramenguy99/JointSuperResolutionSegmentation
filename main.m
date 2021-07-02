clear all;
close all; 

%{
in = imread('./images/black_white_big.png');

constrained    = 1;
alpha_ratio    = 0.004;
mu             = 1e-1;
up_factor      = 2;
blur_variance  = 0.0;
noise_level    = 0.01;

out = FastSegmentation(in, 'constrained', constrained, 'alpha_ratio', alpha_ratio, ...
                       'mu', mu, 'up_factor', up_factor, 'blur_variance', blur_variance,...
                       'noise_level', noise_level);
%}
b = zeros(3,2);
size(b, 1)
size(b, 2)

kernel = fspecial('gaussian', [7 7], 1)

%montage({in, out}, 'size', [1 2]);