function [out] = FastSegmentation(in, varargin)
% Parameters 
ip = inputParser;
addParameter(ip,'constrained', 1);
addParameter(ip,'alpha_ratio', 0.5);
addParameter(ip,'mu', 5e-3);
addParameter(ip,'up_factor', 2);
addParameter(ip,'blur_variance', 1.0);
addParameter(ip,'noise_level', 0.01);
addParameter(ip,'start', 'lanczos');
addParameter(ip,'stop', 'rel_change');
parse(ip, varargin{:});
par = ip.Results;

constrained    = convert_param(par.constrained);
alpha_ratio    = convert_param(par.alpha_ratio);
mu             = convert_param(par.mu);
up_factor      = convert_param(par.up_factor);
blur_variance  = convert_param(par.blur_variance);
noise_level    = convert_param(par.noise_level);
start = par.start;
stop = par.stop;

% Real Image
gt=im2double(in);
gt=modcrop(gt,[up_factor,up_factor]);
HRsize = size(gt);

% Point-Spread-Function and Blur image
if blur_variance > 0
    kernel = fspecial('gaussian', [7 7], blur_variance);
else
    kernel = zeros([7 7]);
    kernel(4, 4) = 1.0;
end

PSFfft = psf2otf(kernel, HRsize);
gt_fft = fft2(gt);
b = real(ifft2(PSFfft .* gt_fft));

% Downsampled image and Noisy image
b   = b(1:up_factor:end, 1:up_factor:end, :);
eta = randn(size(b));
b   = b + noise_level * eta;

% Algorithm parameters
mu_t       = mu;
rel_chg_th = 5e-4;
itr_th     = 200;
alpha      = fix((HRsize(1) * HRsize(2)) * alpha_ratio);
alpha_th   = alpha + alpha * 0.05;

% Starting Point
if strcmp(start,'zeros')
    x = zeros(HRsize);
elseif strcmp(start, 'TV')
    x = imresize(b,up_factor,'lanczos3');
    out = TV0_U_ADMM_FSR(b, kernel, up_factor,x,5e-3,1e-4,itr_th, 0, gt, @TV, 0);
    x = out.x;
    %imshow(out.x);
    %figure;
elseif strcmp(start, 'noise')
    x = rand(HRsize);
else
    x = imresize(b,up_factor,'lanczos3');
end

if strcmp(stop, 'l0')
    stop_l0 = 1;
    rel_chg_th = alpha_th;
else
    stop_l0 = 0;
end
    

debug = 0;
if constrained
     [res] = L0P_U_ADMM_FSR(b, kernel,up_factor,x,alpha_th,itr_th, debug, gt, @L0, alpha);
else
     [res] = TV0_U_ADMM_FSR(b, kernel,up_factor,x,mu_t,rel_chg_th,itr_th, debug, gt, @TV0I, stop_l0);
end

out.x = res.x;
out.J_fids = res.J_fids;
out.J_regs = res.J_regs;
out.l0 = res.l0;
out.b = b;
out.itr = res.itr;
fprintf("itr: %d | L0: %d\n", res.itr, res.l0(end));

end

function [out]=convert_param(in)
    if ~isa(in,'numeric')
        out = str2num(in);
    else
        out = in;
    end
end

function [t, s] = L0(q_h, q_v, alpha)
    [m, n, nc] = size(q_h);
    
    t = reshape(q_h, [m * n, nc]);
    s = reshape(q_v, [m * n, nc]);
    
    A = sum(t.^2, 2) + sum(s.^2, 2);
    N = numel(A);
    
    if N > alpha
        [~, mask] = mink(A, N - alpha);

        t(mask, :) = 0;
        s(mask, :) = 0;
    end
    
    t = reshape(t, [m, n, nc]);
    s = reshape(s, [m, n, nc]);
end

function [t, s] = TV0I(q_h, q_v, mu_t, beta_t)
    [m, n, nc] = size(q_h);
    
    t = reshape(q_h, [m * n, nc]);
    s = reshape(q_v, [m * n, nc]);
    
    mask = sum(t.^2, 2) + sum(s.^2, 2) < ((2*mu_t) / (beta_t));
    t(mask, :) = 0;
    s(mask, :) = 0;
    
    t = reshape(t, [m, n, nc]);
    s = reshape(s, [m, n, nc]);
end


function [t, s] = TV(q_h, q_v, mu_t, beta_t)
    one_over_beta_t = 1.0 / beta_t;
    q_norm              = sqrt( q_h.^2 + q_v.^2 );
    q_norm(q_norm == 0) = mu_t*one_over_beta_t;
    q_norm              = max( q_norm - mu_t*one_over_beta_t , 0 ) ./ q_norm;
    t                   = q_norm .* q_h;
    s                   = q_norm .* q_v;
end