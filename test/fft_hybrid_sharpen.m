function I_out = fft_hybrid_sharpen(I_in, k1, sigma_f, k2, n, D0)
%FFT_HYBRID_SHARPEN
%   Hybrid sharpening = Gaussian High-Boost + Butterworth High-Boost
%
%   INPUTS:
%       I_in    : input grayscale image
%       k1      : Gaussian high-boost gain        (0.5 - 2)
%       sigma_f : Gaussian frequency sigma        (10 - 40)
%       k2      : Butterworth high-boost gain     (0.5 - 2)
%       n       : Butterworth filter order        (1 - 6)
%       D0      : Butterworth cutoff frequency    (10 - 60)
%
%   OUTPUT:
%       I_out   : hybrid sharpened image (uint8)

    I = double(I_in);
    [M, N] = size(I);

    %% === FFT ===
    F = fft2(I);
    F = fftshift(F);

    %% === Frequency Grid (centered) ===
    [u, v] = meshgrid(-floor(N/2):floor((N-1)/2), ...
                      -floor(M/2):floor((M-1)/2));

    D = sqrt(u.^2 + v.^2);


    %% ============================================================
    % 1) Gaussian HIGH-BOOST Filter
    %% ============================================================
    H_gauss = 1 - exp(-(D.^2) / (2 * sigma_f^2));
    H_hb_gauss = 1 + k1 * H_gauss;


    %% ============================================================
    % 2) Butterworth HIGH-BOOST Filter
    %% ============================================================
    H_butt = 1 ./ (1 + (D0 ./ (D + eps)).^(2*n));  % HPF
    H_hb_butt = 1 + k2 * H_butt;


    %% ============================================================
    % 3) HYBRID HIGH-BOOST
    %% ============================================================
    % Weighted geometric mean (stable, smooth)
    H_hybrid = sqrt(H_hb_gauss .* H_hb_butt);


    %% ============================================================
    % 4) Apply to FFT
    %% ============================================================
    F_filtered = F .* H_hybrid;

    %% Inverse FFT
    F_filtered = ifftshift(F_filtered);
    I_f = real(ifft2(F_filtered));

    %% Clip + Output
    I_out = uint8(max(0, min(255, I_f)));
end
