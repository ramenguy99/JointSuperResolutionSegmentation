function [out] = D_L0P_U_ADMM_FSR(b, blur_k,up_factor,x,alpha_th,itr_th, debug, x_true, TV, alpha)
        
% LR and HR dimensions
        size_b(1)  = size(b, 1);
        size_b(2)  = size(b, 2);
        nc = size(b, 3);
        HR_size = size_b(1:2) * up_factor;
        HR_size_rgb = [HR_size, nc];

% Tolerance for the computation of L0 norm
        tol0 = 1.0 / 255;
        %tol0 = 1e-3;% / 255;
        
% Initialize lagrange multipliers
        lambda_t      = zeros(HR_size);
        lambda_s      = zeros(HR_size);
        
% Common matrices
        K_DFT                   = psf2otf(blur_k, HR_size);
        KT_DFT                  = conj(K_DFT);
        K2                      = abs(K_DFT).^2;
        STb                     = zeros(HR_size_rgb);
        STb(1:up_factor:end,1:up_factor:end, :) = b;
        KTSTb                   = real(ifft2(KT_DFT.*fft2(STb)));
        Dh_DFT                  = psf2otf([1,-1],HR_size);
        Dv_DFT                  = psf2otf([1;-1],HR_size);
        c                       = 1e-8;
        F2D                     = abs(Dh_DFT).^2 + abs(Dv_DFT).^2 + c;
        
        Dhx = circshift(x, [ 0, -1]) - x;
        Dvx = circshift(x, [-1,  0]) - x;

        % compute/initialize and store iteration-based quantities
        res                 = real(ifft2(K_DFT .* fft2(x)));
        res                 = res(1:up_factor:end, 1:up_factor:end, :) - b;
        out.J_regs          = nnz( ( sum(abs(Dhx), 3) + (sum(abs(Dvx), 3)) ) > tol0 ) - alpha;
        out.J_fids          = sum( res(:).^2 );
        out.rel_chgs        = 0;
        out.res_means       = mean( res(:) );
        out.res_stdvs       = sqrt( mean( res(:).^2 ) );
        out.relerrs         = norm(sum(abs(x - x_true), 3), 'fro') / norm( sum(abs(x_true), 3), 'fro');
        out.SSIM            = ssim(x_true,x);
        out.PSNR            = psnr(x_true,x);
        % Matlab command window debugs
    if ( debug == 1 )
        ALG_NAME = 'L0P-ADMM-SR';
        fprintf('\n\n')
        fprintf('\n-----------------------------------------------');
        fprintf('\n%s (alpha = %d):',ALG_NAME, alpha);
        fprintf('\n-----------------------------------------------');
        fprintf('\n');
        fprintf('%s  it%04d:   REL-CHG = %15.13f | L0 - \x03B1 = %6d | Jfid = %13.8f | REL-ERR = %10.7f\n',ALG_NAME, 0, out.rel_chgs(end), out.J_regs(end), out.J_fids(end), out.relerrs(end));
    end

% -----------------------------------------------------------------------
% carry out ADMM iterations
% -----------------------------------------------------------------------

% initialize iteration index and stopping criteria flags
itr         = 0;   
stop_flags  = [0,0];

out.l0 = nnz( (sum(abs(Dhx), 3) + (sum(abs(Dvx), 3)) ) > tol0);

while ( sum(stop_flags) == 0 )
    
    % update iteration index
    itr = itr + 1;
    
    % update beta_t
    eps = 1e-3;
    beta_t = itr^(1 + eps);
    
    % ----------------------------------------------------------------
    % solve the ADMM sub-problem for the primal variable t and s (closed form)
    % ----------------------------------------------------------------       
    % First min-subproblem: finding s,t from a closed form 
            q_h = Dhx + lambda_t;
            q_v = Dvx + lambda_s;
            [t, s] = TV(q_h, q_v, alpha);
    
  % --------------------------------------------------------------------
  % solve the ADMM sub-problem for the primal variable x (linear system)
  % --------------------------------------------------------------------   
    
            x_old = x; 
            
            tl = t - lambda_t;
            sl = s - lambda_s;
            b1 = (beta_t) * (circshift(tl, [0, 1]) - tl);
            b2 = (beta_t) * (circshift(sl, [1, 0]) - sl);
            b3 = KTSTb;        

            bb = fft2(b1 + b2 + b3);  

            for i = 1:nc
                x(:,:,i) = INVLS(K_DFT,KT_DFT,K2,bb(:,:,i),(beta_t),(up_factor^2),size_b(1),size_b(2),size_b(1)*size_b(2),F2D);
            end
            
        
    % --------------------------------------------------
    % compute x relative change (for stopping criterion)
    % --------------------------------------------------
    rel_chg = norm(sum(abs(x - x_old), 3), 'fro') / norm( sum(abs(x_old), 3), 'fro');
   
    
    % -------------------------------
    % compute quantities used for the 
    % subsequent lambdas update step
    % and for the next iteration
    % -------------------------------
    
        % gradient of the current iterate
        %[Dx_h,Dx_v] = D(x);
        Dhx = circshift(x, [ 0, -1]) - x;
        Dvx = circshift(x, [-1,  0]) - x;
        
        % Store l0 norm of gradient of the current iterate
        out.l0(end + 1) = nnz( (sum(abs(Dhx), 3) + (sum(abs(Dvx), 3)) ) > tol0);
    
        
    % -----------------------
    % check stopping criteria
    % -----------------------
    if ( itr == itr_th )
        stop_flags(1) = 1;
    end
    if ( out.l0(end) <= alpha_th )
        stop_flags(2) = 1;
    end
    
    % ------------------------------------------------------------------
    % update dual variables (Lagrange multipliers) lambdas (dual ascent)
    % ------------------------------------------------------------------  
        % Lagrange multipliers associated with the auxiliary variable t
        lambda_t      = lambda_t - (t - Dhx);
        lambda_s      = lambda_s - (s - Dvx);
        
    % --------------------------------------
    % (eventually) compute/store/show debugs
    % --------------------------------------
    if ( debug == 1 )
        % compute and store iteration-based quantities
        res                     = real(ifft2(K_DFT .* fft2(x)));
        res                     = res(1:up_factor:end, 1:up_factor:end, :) - b;
        out.J_fids(end+1)       = sum( res(:).^2 );
        out.J_regs(end+1)       = nnz( ( sum(abs(Dhx), 3) + (sum(abs(Dvx), 3)) ) > tol0 ) - alpha;
        out.rel_chgs(end+1)     = rel_chg;
        %out.res_means(end+1)    = mean( res(:) );
        %out.res_stdvs(end+1)    = sqrt( mean( res(:).^2 ) );
        out.relerrs(end+1)      = norm(sum(abs(x - x_true), 3), 'fro') / norm( sum(abs(x_true), 3), 'fro');
        %out.SSIM(end+1)         = ssim(x_true, x);
        %out.PSNR(end+1)         = psnr(x_true, x);
        % Matlab command window debugs
        fprintf('%s  it%04d:   REL-CHG = %15.13f | L0 - \x03B1 = %6d | Jfid = %13.8f | REL-ERR = %10.7f\n',ALG_NAME, itr, out.rel_chgs(end), out.J_regs(end), out.J_fids(end), out.relerrs(end));        
    end
    
end

x=max(min(x,1),0);

% one of the two stopping criteria satisfied --> end iterations and store output results
out.x       = x;
out.itr     = itr;
    
end
