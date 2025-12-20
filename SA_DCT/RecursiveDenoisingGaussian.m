function [y_hat, sigma_est, y_hats] = RecursiveDenoisingGaussian(z, varargin)
% RecursiveDenoisingGaussian - Recursive Anisotropic LPA-ICI Denoising
%
% SYNTAX:
%   y_hat = RecursiveDenoisingGaussian(z)
%   y_hat = RecursiveDenoisingGaussian(z, 'ParameterName', ParameterValue, ...)
%   [y_hat, sigma_est] = RecursiveDenoisingGaussian(...)
%   [y_hat, sigma_est, y_hats] = RecursiveDenoisingGaussian(...)
%
% INPUTS:
%   z           - Noisy input image (grayscale, double [0,1] or uint8 [0,255])
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'sigma'     - Noise standard deviation. If not provided, it will be
%                 automatically estimated using MAD estimator. (default: auto)
%   'quality'   - 'high' for high-quality settings, 'fast' for faster but
%                 lower quality. (default: 'high')
%   'niter'     - Number of iterations (default: auto-selected based on noise)
%   'ndir'      - Number of directions (default: 16 for high, 8 for fast)
%   'gammaICI'  - ICI Gamma threshold (default: 0.8 for high, 1.0 for fast)
%   'verbose'   - Display progress information (default: false)
%
% OUTPUTS:
%   y_hat       - Denoised image
%   sigma_est   - Estimated noise standard deviation
%   y_hats      - Cell array containing intermediate estimates from each iteration
%
% DESCRIPTION:
%   This function performs recursive anisotropic LPA-ICI denoising on 
%   observations contaminated by additive Gaussian white noise.
%
%   Observation model:
%       z = y + n
%   where z is the noisy observation, y is the true image (unknown), 
%   and n is Gaussian white noise.
%
% REFERENCE:
%   Foi, A., V. Katkovnik, K. Egiazarian, and J. Astola,
%   "A novel anisotropic local polynomial estimator based on directional 
%   multiscale optimizations", Proc. of the 6th IMA Int. Conf. Math. in 
%   Signal Processing, Cirencester (UK), pp. 79-82, 2004.
%
% EXAMPLE:
%   % Read and denoise an image
%   z = im2double(imread('noisy_image.png'));
%   y_hat = RecursiveDenoisingGaussian(z);
%
%   % With custom parameters
%   y_hat = RecursiveDenoisingGaussian(z, 'quality', 'fast', 'verbose', true);
%
%   % With known noise level
%   y_hat = RecursiveDenoisingGaussian(z, 'sigma', 25/255);
%
% Alessandro Foi - Tampere University of Technology - 2003-2016
% Function wrapper created for convenient usage

    %----------------------------------------------------------------------
    % Parse input arguments
    %----------------------------------------------------------------------
    p = inputParser;
    addRequired(p, 'z', @(x) isnumeric(x) && ismatrix(x));
    addParameter(p, 'sigma', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'quality', 'high', @(x) ismember(lower(x), {'high', 'fast'}));
    addParameter(p, 'niter', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
    addParameter(p, 'ndir', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 4));
    addParameter(p, 'gammaICI', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
    addParameter(p, 'verbose', false, @islogical);
    
    parse(p, z, varargin{:});
    opts = p.Results;
    
    %----------------------------------------------------------------------
    % Convert input to double if necessary
    %----------------------------------------------------------------------
    if isa(z, 'uint8')
        z = im2double(z);
    elseif ~isa(z, 'double')
        z = double(z);
    end
    
    %----------------------------------------------------------------------
    % Select quality settings
    %----------------------------------------------------------------------
    if strcmpi(opts.quality, 'fast')
        % FAST, LOWER-QUALITY SETTINGS
        h1 = [1 2 3 5];                   % LPA Kernels' length
        sharparams = -1;                  % -1 zero order
        gammaICI = 1;                     % ICI Gamma threshold
        ndir = 8;                         % number of directions
        niter_default = 6;                % number of iterations
        restrict = 0;                     % restricts number of iterations if noise level is small
    else
        % HIGH-QUALITY SETTINGS
        h1 = [1 2 3 5 7 10];              % LPA Kernels' length
        sharparams = [-0.8 -0.6 -0.3];    % sharpening parameters
        gammaICI = 0.8;                   % ICI Gamma threshold
        ndir = 16;                        % number of directions
        niter_default = 3;                % number of iterations
        restrict = 1;                     % restricts number of iterations if noise level is small
    end
    
    % Override with user-specified values
    if ~isempty(opts.ndir)
        ndir = opts.ndir;
    end
    if ~isempty(opts.gammaICI)
        gammaICI = opts.gammaICI;
    end
    
    %----------------------------------------------------------------------
    % Fixed algorithm parameters
    %----------------------------------------------------------------------
    fusing = 1;                         % fusing type (1 classical fusing)
    iterrule = 1;                       % sigmaiter rule
    itercoef = 2/3;                     % compensating factor for recursive std
    
    %----------------------------------------------------------------------
    % LPA kernel parameters
    %----------------------------------------------------------------------
    h2 = ones(size(h1));
    lenh = length(h1);
    window_type = 1;                    % uniform window
    TYPE = 10;                          % NONSYMMETRIC ON X1 and SYMMETRIC ON X2
    
    %----------------------------------------------------------------------
    % Get image size
    %----------------------------------------------------------------------
    [size_z_1, size_z_2] = size(z);
    
    %----------------------------------------------------------------------
    % Estimate noise standard deviation if not provided
    %----------------------------------------------------------------------
    if isempty(opts.sigma)
        sigma = function_stdEst(z);
        if opts.verbose
            fprintf('Estimated noise sigma: %.4f (%.2f on 0-255 scale)\n', sigma, sigma*255);
        end
    else
        sigma = opts.sigma;
        if opts.verbose
            fprintf('Using provided noise sigma: %.4f\n', sigma);
        end
    end
    sigma_est = sigma;
    
    %----------------------------------------------------------------------
    % Determine number of iterations
    %----------------------------------------------------------------------
    if ~isempty(opts.niter)
        niter = opts.niter;
    else
        niter = niter_default;
        if restrict == 1
            % Restrict number of iterations when noise is not too large
            niter = max(1, min(round(sigma * 43), niter));
        end
    end
    
    if opts.verbose
        fprintf('Running %d iterations with %d directions\n', niter, ndir);
    end
    
    %----------------------------------------------------------------------
    % Create LPA kernels
    %----------------------------------------------------------------------
    [kernels, kernels_higher_order] = function_CreateLPAKernels([0 0], h1, h2, TYPE, window_type, ndir, ones(2,lenh), 1);
    [~, kernels_higher_orderb] = function_CreateLPAKernels([1 0], h1, h2, TYPE, window_type, ndir, ones(2,lenh), 1);
    
    if opts.verbose
        fprintf('LPA kernels created\n');
    end
    
    %----------------------------------------------------------------------
    % Initialize variables
    %----------------------------------------------------------------------
    z_iter = z;
    sigmaiter = repmat(sigma, size_z_1, size_z_2);
    y_hats = cell(1, niter);
    
    %----------------------------------------------------------------------
    % Main recursive loop
    %----------------------------------------------------------------------
    for momo = 1:niter
        sharparam = sharparams(min(momo, numel(sharparams)));
        
        % Initialize fusing variables
        YICI_Final1 = 0;
        var_inv = 0;
        CWW = 0;
        CWW2 = 0;
        
        % Preallocate
        yh = zeros(size_z_1, size_z_2, lenh);
        stdh = zeros(size_z_1, size_z_2, lenh);
        ghorigin = zeros(ndir, lenh);
        YICI_Q = zeros(size_z_1, size_z_2, ndir);
        var_opt_Q = zeros(size_z_1, size_z_2, ndir);
        
        % Loop over directions
        for s1 = 1:ndir
            % Loop over kernel sizes
            for s2 = 1:lenh
                gha = kernels_higher_order{s1, s2, 1}(:,:,1);
                ghb = kernels_higher_orderb{s1, s2, 1}(:,:,1);
                gh = (1 + sharparam) * ghb - sharparam * gha;
                
                % Remove unnecessary zeroes
                bound1 = min([find(sum(gh~=0, 2)); abs(find(sum(gh~=0, 2)) - size(gh,1) - 1)]);
                bound2 = min([find(sum(gh~=0, 1)), abs(find(sum(gh~=0, 1)) - size(gh,2) - 1)]);
                gh = gh(bound1:size(gh,1)-bound1+1, bound2:size(gh,2)-bound2+1);
                
                ghorigin(s1, s2) = gh((end+1)/2, (end+1)/2);
                
                % Estimation
                yh(:,:,s2) = conv2(z_iter + 10000, gh, 'same') - 10000;
                
                % Standard deviation of the estimate
                if momo == 1
                    stdh(:,:,s2) = repmat(sigma * sqrt(sum(gh(:).^2)), size_z_1, size_z_2);
                else
                    stdh(:,:,s2) = sqrt(conv2(sigmaiter.^2, gh.^2, 'same'));
                end
            end
            
            % ICI rule
            [YICI, h_opt, std_opt] = function_ICI(yh, stdh, gammaICI, 2*(s1-1)*pi/ndir);
            
            % Origin weight for optimal kernels
            aaa = reshape(ghorigin(s1, h_opt), size(h_opt));
            
            YICI_Q(:,:,s1) = YICI;
            var_opt_Q(:,:,s1) = std_opt.^2 + eps;
            
            % Fusing
            YICI_Final1 = YICI_Final1 + YICI_Q(:,:,s1) ./ var_opt_Q(:,:,s1);
            var_inv = var_inv + 1 ./ var_opt_Q(:,:,s1);
            CWW = CWW + aaa ./ var_opt_Q(:,:,s1);
            CWW2 = CWW2 + (aaa ./ var_opt_Q(:,:,s1)).^2;
        end
        
        % Final fused estimate
        if fusing == 1
            y_hat = YICI_Final1 ./ var_inv;
        end
        
        % Store result
        y_hats{momo} = y_hat;
        
        if opts.verbose
            fprintf('Iteration %d/%d completed\n', momo, niter);
        end
        
        % Update standard deviation for next iteration
        if momo < niter
            sigmaiter = sqrt((1 ./ (var_inv.^2)) .* (var_inv - (sigmaiter.^2) .* CWW2 + (sigmaiter .* CWW).^2));
            sigmaiter = sigmaiter * itercoef;
            
            % Update observation for next iteration
            z_iter = y_hat;
        end
    end
    
    if opts.verbose
        fprintf('Denoising completed\n');
    end
end
