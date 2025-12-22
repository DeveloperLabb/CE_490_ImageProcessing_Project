function [y_hat, out] = deblur_LPAICI_Gaussian(z_in, v, varargin)
%DEBLUR_LPAICI_GAUSSIAN  Anisotropic LPA-ICI deconvolution (RI + optional RW)
%
%   [y_hat, out] = deblur_LPAICI_Gaussian(z_in, v, 'Name', Value, ...)
%
%   INPUTS
%     z_in : blurred (and possibly noisy) observation image (grayscale or RGB)
%            uint8/uint16/single/double supported
%     v    : PSF kernel (2D). If [] or omitted, defaults to 9x9 box PSF.
%
%   NAME-VALUE OPTIONS (all optional)
%     'EstimateSigma'        : true/false (default true)
%     'Sigma'                : noise sigma (used if EstimateSigma=false or provided)
%     'DoWiener'             : true/false (default true)  % RW stage
%     'EstimateDerivative'   : true/false (default true)  % only if DoWiener=true
%     'RegularizationRI'     : epsilon for RI  (default 0.014)
%     'RegularizationRW'     : epsilon for RW  (default 0.11)
%     'GammaRI'              : ICI gamma RI (default 1.35)
%     'GammaRW'              : ICI gamma RW (default 1.25)
%     'AlphaOrderRI'         : -1 (0th order) or 0 (1st order) (default -1)
%     'AlphaOrderRW'         : -1 or 0 (default -1)
%     'Precooked'            : 1/0 (default 1)  % optimized kernels (requires ndirRI=8 or 4)
%     'ShowFigures'          : true/false (default false)
%     'Verbose'              : true/false (default true)
%     'ClipTo01'             : true/false (default true)
%
%   OUTPUTS
%     y_hat : final deblurred image (RW result if DoWiener=true, else RI result)
%     out   : struct with intermediate outputs if available (y_hat_RI, y_hat_RW, etc.)
%
%   NOTES
%     - This function calls the original working programs:
%         function_DeblurringGaussian_RI  (script)
%         function_DeblurringGaussian_RW  (script)
%       They must be on the MATLAB path and compatible with being run inside a function.
%

    if nargin < 2 || isempty(v)
        v = ones(9); v = v ./ sum(v(:)); % default PSF (Exp #1 style)
    end

    % -------------------------
    % Parse options
    % -------------------------
    p = inputParser;
    p.addParameter('EstimateSigma', true, @(x)islogical(x) || isnumeric(x));
    p.addParameter('Sigma', [], @(x)isnumeric(x) && (isempty(x) || isscalar(x)));
    p.addParameter('DoWiener', true, @(x)islogical(x) || isnumeric(x));
    p.addParameter('EstimateDerivative', true, @(x)islogical(x) || isnumeric(x));

    p.addParameter('RegularizationRI', 0.014, @(x)isnumeric(x) && isscalar(x) && x>0);
    p.addParameter('RegularizationRW', 0.11,  @(x)isnumeric(x) && isscalar(x) && x>0);

    p.addParameter('GammaRI', 1.35, @(x)isnumeric(x) && isscalar(x) && x>0);
    p.addParameter('GammaRW', 1.25, @(x)isnumeric(x) && isscalar(x) && x>0);

    p.addParameter('AlphaOrderRI', -1, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('AlphaOrderRW', -1, @(x)isnumeric(x) && isscalar(x));

    p.addParameter('Precooked', 1, @(x)isnumeric(x) && isscalar(x));
    p.addParameter('ShowFigures', false, @(x)islogical(x) || isnumeric(x));
    p.addParameter('Verbose', true, @(x)islogical(x) || isnumeric(x));
    p.addParameter('ClipTo01', true, @(x)islogical(x) || isnumeric(x));

    p.parse(varargin{:});
    opts = p.Results;

    opts.EstimateSigma      = logical(opts.EstimateSigma);
    opts.DoWiener           = logical(opts.DoWiener);
    opts.EstimateDerivative = logical(opts.EstimateDerivative);
    opts.ShowFigures        = logical(opts.ShowFigures);
    opts.Verbose            = logical(opts.Verbose);
    opts.ClipTo01           = logical(opts.ClipTo01);

    % -------------------------
    % Convert input to double
    % -------------------------
    z_in = normalizeToDouble01(z_in);

    % -------------------------
    % Process grayscale or RGB
    % -------------------------
    if ndims(z_in) == 2
        [y_hat, out] = deblurOneChannel(z_in, v, opts);
    elseif ndims(z_in) == 3 && size(z_in,3) == 3
        y_hat = zeros(size(z_in), 'like', z_in);
        out = struct();
        out.channels = cell(1,3);
        for c = 1:3
            [y_hat(:,:,c), out.channels{c}] = deblurOneChannel(z_in(:,:,c), v, opts);
        end
    else
        error('z_in must be 2D grayscale or 3-channel RGB.');
    end

    if opts.ClipTo01
        y_hat = min(max(y_hat, 0), 1);
    end
end

% =====================================================================
% Deblur one channel (runs the same RI/RW scripts in this workspace)
% =====================================================================
function [y_hat, out] = deblurOneChannel(z, v, opts)

    % ------------------------------------------------------------------
    % Algorithm parameters (kept consistent with the demo defaults)
    % ------------------------------------------------------------------
    estimate_sigma      = opts.EstimateSigma;
    do_wiener           = opts.DoWiener;
    estimate_derivative = opts.EstimateDerivative;

    % LPA windows
    h1RI  = [1 3 5 6 11];
    h1RW  = [1 3 5 8 17];
    h2RI  = ones(size(h1RI));
    h2RW  = max(1, ceil(h1RW*tan(0.5*pi/8)));
    ndirRI = 8;
    ndirRW = 8;
    
    % >>> ADD THESE (needed by RW/RI scripts)
    lenhRI = length(h2RI);
    lenhRW = length(h1RW);

    % ICI thresholds
    GammaParameterRI = opts.GammaRI;
    GammaParameterRW = opts.GammaRW;

    % LPA order-mixture parameter
    alphaorderRI = opts.AlphaOrderRI;
    alphaorderRW = opts.AlphaOrderRW;

    Precooked = opts.Precooked;

    % Regularization parameters
    Regularization_epsilon_RI = opts.RegularizationRI;
    Regularization_epsilon_RW = opts.RegularizationRW;

    % ------------------------------------------------------------------
    % Build FFT of PSF over image domain (the scripts expect V in workspace)
    % ------------------------------------------------------------------
    [yN, xN] = size(z);
    size_z_1 = yN;
    size_z_2 = xN;

    [ghy, ghx] = size(v);
    big_v = zeros(yN, xN);
    big_v(1:ghy, 1:ghx) = v;
    big_v = circshift(big_v, -round([(ghy-1)/2, (ghx-1)/2]));
    V = fft2(big_v);

    % ------------------------------------------------------------------
    % Noise sigma (either given or estimated)
    % ------------------------------------------------------------------
    if ~isempty(opts.Sigma)
        sigma = opts.Sigma;
    else
        if estimate_sigma
            sigma = function_stdEst(z);
        else
            error('Sigma must be provided if EstimateSigma=false.');
        end
    end

    if opts.Verbose
        if estimate_sigma && isempty(opts.Sigma)
            disp(['Estimated noise sigma = ', num2str(sigma)]);
        else
            disp(['Using noise sigma = ', num2str(sigma)]);
        end
    end

    % ------------------------------------------------------------------
    % Run RI stage (script)
    % ------------------------------------------------------------------
    if opts.Verbose, disp('starting RI ...'); end
    function_DeblurringGaussian_RI;   % produces: zRI, y_hat_RI, h_optRI, etc.

    % ------------------------------------------------------------------
    % Run RW stage (optional) (script)
    % ------------------------------------------------------------------
    if do_wiener
        if opts.Verbose, disp('starting RW ...'); end
        Wiener_Pilot = abs(fft2(y_hat_RI));
        function_DeblurringGaussian_RW; % produces: zRW, y_hat_RW, h_opt_Q, yder, etc.
        y_hat = y_hat_RW;
    else
        y_hat = y_hat_RI;
    end

    % ------------------------------------------------------------------
    % Optional figures
    % ------------------------------------------------------------------
    if opts.ShowFigures
        figure; imshow(z, []); title('Input observation z');
        figure; imshow(y_hat_RI, []); title('RI estimate');
        if do_wiener
            figure; imshow(y_hat_RW, []); title('RW (final) estimate');
        end
    end

    % ------------------------------------------------------------------
    % Collect outputs safely (only if they exist)
    % ------------------------------------------------------------------
    out = struct();
    out.sigma = sigma;
    out.v = v;

    if exist('y_hat_RI','var'), out.y_hat_RI = y_hat_RI; end
    if exist('zRI','var'),      out.zRI      = zRI;      end
    if exist('h_optRI','var'),  out.h_optRI  = h_optRI;  end

    if do_wiener
        if exist('y_hat_RW','var'), out.y_hat_RW = y_hat_RW; end
        if exist('zRW','var'),      out.zRW      = zRW;      end
        if exist('h_opt_Q','var'),  out.h_opt_Q  = h_opt_Q;  end
        if exist('yder','var'),     out.yder     = yder;     end
        if exist('yd_hat_Q','var'), out.yd_hat_Q = yd_hat_Q; end
    end
end

% =====================================================================
% Helper: convert to double [0,1] if integer; keep double/single as-is
% =====================================================================
function I = normalizeToDouble01(Iin)
    if isa(Iin, 'uint8') || isa(Iin, 'uint16') || isa(Iin, 'uint32')
        I = im2double(Iin);
    elseif isa(Iin, 'single')
        I = double(Iin);
    else
        I = double(Iin);
    end
end
