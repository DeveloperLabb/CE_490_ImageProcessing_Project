function [Yhat, info] = sa_dct_blind_deblurring(Zin)
%SADCT_BLIND_DEBLUR  SA-DCT deblur+denoise with UNKNOWN PSF,
%but here we assume PSF is symmetric Gaussian with std ~ [1,2].
%
%   [Yhat, info] = sadct_blind_deblur(Zin)
%
% INPUT:
%   Zin : blurred+noisy image (uint8/uint16/single/double), grayscale or RGB.
%
% OUTPUT:
%   Yhat : restored image (same class as Zin)
%   info : struct (sigma, psf_est, psf_sigma, eps_RI/RW, chosen_mode, scores)
%
% REQUIREMENTS:
%   SA-DCT toolbox functions on path:
%     function_CreateLPAKernels, function_ICI,
%     function_SADCT_RI_thresholding, function_SADCT_RW_wiener
%   Optional: niqe (Image Processing Toolbox) for better no-ref scoring
%   Optional: RecursiveDenoisingGaussian (SA-DCT denoise-only baseline)

% -----------------------------
% 0) Input -> double in [0,1]
% -----------------------------
in_class = class(Zin);
Z = Zin;
if ndims(Z) == 3
    Z = rgb2gray(Z);
end

Z = double(Z);
if isa(Zin,'uint8')
    Z = Z/255;
elseif isa(Zin,'uint16')
    Z = Z/65535;
end
Z = min(max(Z,0),1);

[H,W] = size(Z);
N = H*W;

% -----------------------------
% 1) Sigma estimate (prefer SA-DCT demo estimator)
% -----------------------------
if exist('function_stdEst','file') == 2
    sigma = function_stdEst(Z);
else
    sigma = local_sigma_mad_highpass(Z);
end
sigma = max(sigma, 1e-6);

% -----------------------------
% 2) SAFE baseline: SA-DCT denoise-only (so we never return worse)
% -----------------------------
Y_denoise = Z;
if exist('RecursiveDenoisingGaussian','file') == 2
    try
        Y_denoise = RecursiveDenoisingGaussian(Z);
        Y_denoise = min(max(Y_denoise,0),1);
    catch
        Y_denoise = Z;
    end
end

useNIQE = exist('niqe','file') == 2;
if useNIQE
    score_denoise = niqe(uint8(255*Y_denoise));
else
    score_denoise = local_noref_score(Y_denoise);
end

% -----------------------------
% 3) PSF ESTIMATION (Gaussian-only, std in [1,2])
%    We pick sigma_psf by scoring a quick conservative inverse preview.
% -----------------------------
sigma_psf_list = 1.0:0.1:2.0;

bestScore = inf;
bestPSF = [];
bestSigmaPSF = NaN;

for sg = sigma_psf_list
    psf_size = 2*ceil(3*sg)+1;      % covers ~±3σ
    psf_size = max(psf_size, 7);
    psf_size = min(psf_size, 15);   % keep small/fast and realistic
    if mod(psf_size,2) == 0, psf_size = psf_size + 1; end

    v = fspecial('gaussian', [psf_size psf_size], sg);
    v = v ./ sum(v(:));

    % Centered FFT of PSF
    big_v = zeros(H,W);
    big_v(1:psf_size,1:psf_size) = v;
    big_v = circshift(big_v, -round([(psf_size-1)/2 (psf_size-1)/2]));
    V = fft2(big_v);

    % Quick conservative inverse preview (NOT full SA-DCT yet)
    % Use a fixed moderate epsilon to avoid selecting too-aggressive PSF.
    eps_preview = 0.10;
    Yf = fft2(Z) .* conj(V) ./ (abs(V).^2 + eps_preview^2);
    y_preview = real(ifft2(Yf));
    y_preview = min(max(y_preview,0),1);

    % Score preview (lower is better)
    if useNIQE
        sc = niqe(uint8(255*y_preview));
    else
        sc = local_noref_score(y_preview);
    end

    if sc < bestScore
        bestScore = sc;
        bestPSF = v;
        bestSigmaPSF = sg;
    end
end

% If for some reason we didn't set, fallback to sg=1.5
if isempty(bestPSF)
    bestSigmaPSF = 1.5;
    psf_size = 2*ceil(3*bestSigmaPSF)+1;
    psf_size = max(psf_size, 7);
    psf_size = min(psf_size, 15);
    if mod(psf_size,2) == 0, psf_size = psf_size + 1; end
    bestPSF = fspecial('gaussian', [psf_size psf_size], bestSigmaPSF);
    bestPSF = bestPSF ./ sum(bestPSF(:));
end

v = bestPSF;

% -----------------------------
% 4) SA-DCT deconvolution stages (conservative + safe)
% -----------------------------
GammaParameterRI = 0.9;
DCTthrCOEF       = 0.88;
GammaParameterRW = 1.7;
DCTwieCOEF       = 1.0;

h1RI = [1 2 4];
h1RW = [1 2 4];
max_overlap  = 49;
max_overlapW = 49;
alphaorderRI = -0.7;
alphaorderRW = -0.8;
small_size_RI = 32;
small_size_RW = 32;

% FFT PSF centered
[ghy,ghx] = size(v);
big_v = zeros(H,W);
big_v(1:ghy,1:ghx) = v;
big_v = circshift(big_v, -round([(ghy-1)/2 (ghx-1)/2]));
V = fft2(big_v);

% Choose epsRI with HF-penalized discrepancy (prevents noise blow-up)
epsRI = local_choose_eps_RI_safe(Z, V, sigma);

% --- RI step ---
RI = conj(V) ./ (abs(V).^2 + epsRI^2);
ri = real(ifft2(RI));
zRI = real(ifft2(fft2(Z) .* RI));

% LPA kernels (RI)
h2RI = ones(size(h1RI));
lenhRI = numel(h1RI);
[k0RI, ~] = function_CreateLPAKernels([0,0], h1RI, h2RI, 10, 1, 1, ones(2,lenhRI), 0);
[k1RI, ~] = function_CreateLPAKernels([1,0], h1RI, h2RI, 10, 1, 1, ones(2,lenhRI), 0);

h_max = max(h1RI);
TrianRotu = local_build_triangles(h1RI);

% coeff variance approximation (RI)
small_size = small_size_RI;
tran_fftmatrix = (ifft(small_size*eye(2*h_max-1), small_size))';
tran_fftmatrix_fast = tran_fftmatrix(:,1:end/2+1);
conj_fftmatrix = conj(tran_fftmatrix');
low_pass_k = ones(round(H/small_size), round(W/small_size));
low_pass_k = conv2(low_pass_k, ones(2));
low_pass_k = low_pass_k/(sum(low_pass_k(:)));
RIsmallabs2 = conv2(repmat(abs(RI).^2,[3 3]), low_pass_k, 'same');
RIsmallabs2 = RIsmallabs2(H+1+round((H/small_size)*(0:small_size-1)), ...
                          W+1+round((W/small_size)*(0:small_size-1)));

% anisotropic LPA-ICI (RI)
directional_resolution = 8;
h_opt_Q = zeros(H,W,directional_resolution,'uint8');

yh_RI   = zeros(H,W,lenhRI);
stdh_RI = zeros(H,W,lenhRI);

for s1 = 1:directional_resolution
    for s = 1:lenhRI
        gh = (1+alphaorderRI)*k1RI{1,s} - alphaorderRI*k0RI{1,s};
        if ~rem(s1,2)
            for iziz=1:size(gh,2)
                gh(:,iziz)=circshift(gh(:,iziz),[(size(gh,2)+1)/2-iziz,0]);
            end
        end
        gh = rot90(gh, floor((s1-1)/2));

        yh_RI(:,:,s) = conv2(zRI-100000, gh, 'same') + 100000;

        tmp = conv2(ri([H-h1RI(s)+2:H 1:H 1:h1RI(s)-1], ...
                       [W-h1RI(s)+2:W 1:W 1:h1RI(s)-1]), gh, 'valid');
        stdh_RI(:,:,s) = sqrt(sum(tmp(:).^2))*sigma;
    end
    [~, h_optRI, ~] = function_ICI(yh_RI, stdh_RI, GammaParameterRI, 2*(s1-1)*pi/8);
    h_opt_Q(:,:,s1) = uint8(h_optRI);
end

% SA-DCT threshold schedule
SSS = 1:(2*h_max-1)^2;
T  = DCTthrCOEF * sqrt(2*log(SSS) + 1);

hadper = cell(1, 2*h_max-1);
for h = 1:(2*h_max-1)
    hadper{h} = dct(eye(h));
end

% lexicographic P2P
h_opt_Qd = double(h_opt_Q);
h_opts_matrix = h_opt_Qd(:,:,1);
for iiii=2:directional_resolution
    h_opts_matrix = h_opts_matrix + h_opt_Qd(:,:,iiii)*10^(iiii-1);
end
P2P = [(1:N)', h_opts_matrix(:)];
P2P = P2P(randperm(N),:);
P2P = uint32(sortrows(P2P,2));

y_hat_RI = function_SADCT_RI_thresholding( ...
    h1RI, uint8(h_opt_Q), hadper, TrianRotu, conj_fftmatrix, tran_fftmatrix_fast, ...
    small_size, RIsmallabs2, zRI, P2P, sigma, DCTthrCOEF, T, h_max, max_overlap);

% --- RW step ---
Wiener_Pilot = abs(fft2(y_hat_RI));
epsRW = local_choose_eps_RW_safe(Z, V, Wiener_Pilot, sigma);

PSD = N * sigma^2;
RW = conj(V).*Wiener_Pilot.^2 ./ (Wiener_Pilot.^2.*(abs(V).^2) + PSD*epsRW);
rw = real(ifft2(RW));
zRW = real(ifft2(fft2(Z).*RW));

% LPA kernels (RW)
h2RW = ones(size(h1RW));
lenhRW = numel(h1RW);
[k0RW, ~] = function_CreateLPAKernels([0,0], h1RW, h2RW, 10, 1, 1, ones(2,lenhRW), 0);
[k1RW, ~] = function_CreateLPAKernels([1,0], h1RW, h2RW, 10, 1, 1, ones(2,lenhRW), 0);

h_max = max(h1RW);
TrianRotu = local_build_triangles(h1RW);

% coeff variance approximation (RW)
small_size = small_size_RW;
tran_fftmatrix = (ifft(small_size*eye(2*h_max-1), small_size))';
tran_fftmatrix_fast = tran_fftmatrix(:,1:end/2+1);
conj_fftmatrix = conj(tran_fftmatrix');
low_pass_k = ones(round(H/small_size), round(W/small_size));
low_pass_k = conv2(low_pass_k, ones(2));
low_pass_k = low_pass_k/(sum(low_pass_k(:)));
RWsmallabs2 = conv2(repmat(abs(RW).^2,[3 3]), low_pass_k, 'same');
RWsmallabs2 = RWsmallabs2(H+1+round((H/small_size)*(0:small_size-1)), ...
                          W+1+round((W/small_size)*(0:small_size-1)));

% anisotropic LPA-ICI (RW)
h_opt_Q = zeros(H,W,directional_resolution,'uint8');
yh_RW   = zeros(H,W,lenhRW);
stdh_RW = zeros(H,W,lenhRW);

for s1 = 1:directional_resolution
    for s = 1:lenhRW
        gh = (1+alphaorderRW)*k1RW{1,s} - alphaorderRW*k0RW{1,s};
        if ~rem(s1,2)
            for iziz=1:size(gh,2)
                gh(:,iziz)=circshift(gh(:,iziz),[(size(gh,2)+1)/2-iziz,0]);
            end
        end
        gh = rot90(gh, floor((s1-1)/2));

        yh_RW(:,:,s) = conv2(zRW-100000, gh, 'same') + 100000;

        tmp = conv2(rw([H-h1RW(s)+2:H 1:H 1:h1RW(s)-1], ...
                       [W-h1RW(s)+2:W 1:W 1:h1RW(s)-1]), gh, 'valid');
        stdh_RW(:,:,s) = sqrt(sum(tmp(:).^2))*sigma;
    end
    [~, h_optRW, ~] = function_ICI(yh_RW, stdh_RW, GammaParameterRW, 2*(s1-1)*pi/8);
    h_opt_Q(:,:,s1) = uint8(h_optRW);
end

% DCT bases again (RW)
hadper = cell(1, 2*h_max-1);
for h = 1:(2*h_max-1)
    hadper{h} = dct(eye(h));
end

% lexicographic P2P
h_opt_Qd = double(h_opt_Q);
h_opts_matrix = h_opt_Qd(:,:,1);
for iiii=2:directional_resolution
    h_opts_matrix = h_opts_matrix + h_opt_Qd(:,:,iiii)*10^(iiii-1);
end
P2P = [(1:N)', h_opts_matrix(:)];
P2P = P2P(randperm(N),:);
P2P = uint32(sortrows(P2P,2));

y_hat_RW = function_SADCT_RW_wiener( ...
    h1RW, uint8(h_opt_Q), hadper, TrianRotu, conj_fftmatrix, tran_fftmatrix_fast, ...
    small_size, RWsmallabs2, zRW, P2P, sigma, T, h_max, y_hat_RI, DCTwieCOEF, max_overlapW);

Y_blind = min(max(y_hat_RW,0),1);

% -----------------------------
% 5) Safety selection: return best between denoise-only and deblur
% -----------------------------
if useNIQE
    score_blind = niqe(uint8(255*Y_blind));
else
    score_blind = local_noref_score(Y_blind);
end

if score_blind <= score_denoise
    Y_final = Y_blind;
    chosen_mode = "gaussian_blur_deblur";
    score_final = score_blind;
else
    Y_final = Y_denoise;
    chosen_mode = "denoise_only_safer";
    score_final = score_denoise;
end

% -----------------------------
% Output cast + info
% -----------------------------
Yhat = local_cast_out(Y_final, in_class);

info = struct();
info.sigma = sigma;
info.psf_est = v;
info.psf_model = "gaussian";
info.psf_sigma = bestSigmaPSF;
info.eps_RI = epsRI;
info.eps_RW = epsRW;
info.chosen_mode = chosen_mode;
info.score_denoise = score_denoise;
info.score_blind = score_blind;
info.score_final = score_final;

end

% =========================================================================
% Helpers
% =========================================================================

function sigma = local_sigma_mad_highpass(I)
hp = I - imgaussfilt(I, 1.0);
sigma = median(abs(hp(:))) / 0.6745;
end

function score = local_noref_score(I)
% Lower is better: penalize high-frequency noise strongly
hp = I - imgaussfilt(I, 1.0);
noise = var(hp(:));
L = [0 1 0; 1 -4 1; 0 1 0];
lap = conv2(I, L, 'valid');
sharp = -mean(abs(lap(:)));
score = sharp + 1.25*noise;
end

function epsRI = local_choose_eps_RI_safe(z, V, sigma)
eps_list = logspace(log10(0.02), log10(0.8), 24);
target = sigma^2;

Zf = fft2(z);
absV2 = abs(V).^2;

best = inf; epsRI = eps_list(1);
for e = eps_list
    Yf = Zf .* conj(V) ./ (absV2 + e^2);
    y  = real(ifft2(Yf));
    y  = min(max(y,0),1);

    zhat = real(ifft2(fft2(y).*V));
    r = z - zhat;

    hp = y - imgaussfilt(y, 1.0);
    hfvar = var(hp(:));

    sc = abs(var(r(:)) - target) + 0.75*hfvar;
    if sc < best
        best = sc;
        epsRI = e;
    end
end
end

function epsRW = local_choose_eps_RW_safe(z, V, pilotAbs, sigma)
eps_list = logspace(log10(0.02), log10(0.8), 24);
target = sigma^2;

Zf = fft2(z);
absV2 = abs(V).^2;
pilot2 = pilotAbs.^2;
N = numel(z);
PSD = N * sigma^2;

best = inf; epsRW = eps_list(1);
for e = eps_list
    RW = conj(V).*pilot2 ./ (pilot2.*absV2 + PSD*e);
    y = real(ifft2(Zf .* RW));
    y = min(max(y,0),1);

    zhat = real(ifft2(fft2(y).*V));
    r = z - zhat;

    hp = y - imgaussfilt(y, 1.0);
    hfvar = var(hp(:));

    sc = abs(var(r(:)) - target) + 0.75*hfvar;
    if sc < best
        best = sc;
        epsRW = e;
    end
end
end

function TrianRotu = local_build_triangles(h1)
h_max = max(h1);
Trian = cell(max(h1), max(h1));
TrianRotu = cell(max(h1), max(h1), 8);

for h_opt_1 = h1
    for h_opt_2 = h1
        M = zeros(2*h_max-1);
        for i1 = h_max-h_opt_2+1 : h_max
            for i2 = 2*h_max-i1 : (h_max-1+h_opt_1-(h_max-i1)*((h_opt_1-h_opt_2)/(h_opt_2-1+eps)))
                M(i1,i2) = 1;
            end
        end
        Trian{h_opt_1,h_opt_2} = M;
    end
end

for ii = 1:8
    for h_opt_1 = h1
        for h_opt_2 = h1
            if mod(ii,2) == 0
                TrianRotu{h_opt_1,h_opt_2,ii} = uint8(rot90(Trian{h_opt_2,h_opt_1}', mod(2+floor((ii-1)/2),8)));
            else
                TrianRotu{h_opt_1,h_opt_2,ii} = uint8(rot90(Trian{h_opt_1,h_opt_2},  mod(floor((ii-1)/2),8)));
            end
        end
    end
end
end

function Yout = local_cast_out(Y01, in_class)
Y01 = min(max(Y01,0),1);
switch in_class
    case 'uint8'
        Yout = uint8(round(255*Y01));
    case 'uint16'
        Yout = uint16(round(65535*Y01));
    otherwise
        Yout = cast(Y01, in_class);
end
end
