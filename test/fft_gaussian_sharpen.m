function I_out = fft_gaussian_sharpen(I_in, k, sigma_f)
%FFT_GAUSSIAN_SHARPEN (with symmetric padding)
%   Frequency-domain Gaussian high-boost sharpening with
%   dimension optimization + symmetric padding.
%
%   I_out = fft_gaussian_sharpen(I_in, k, sigma_f)
%
%   k       : high-boost gain (0.5–2)
%   sigma_f : frequency-domain Gaussian sigma (10–40)

    I = double(I_in);
    [M, N] = size(I);

    %% --- Dimension optimization for FFT ---
    P = 2^nextpow2(M);
    Q = 2^nextpow2(N);

    %% --- Symmetric padding instead of zero padding ---
    padM = P - M;
    padN = Q - N;

    I_pad = padarray(I, [padM padN], "symmetric", "post");

    %% --- Forward FFT ---
    F = fft2(I_pad);

    %% --- Construct Gaussian Low-Pass Filter in Frequency Domain ---
    [u, v] = meshgrid(0:Q-1, 0:P-1);

    u = u - floor(Q/2);
    v = v - floor(P/2);

    D2 = u.^2 + v.^2;

    H_lp = exp(-D2 / (2 * sigma_f^2));   % Gaussian LPF
    H_lp = fftshift(H_lp);

    %% --- High-Boost Filter ---
    H_hb = 1 + k * (1 - H_lp);

    %% --- Apply Frequency Filter ---
    F_filtered = F .* H_hb;

    %% --- Inverse FFT ---
    I_filt = real(ifft2(F_filtered));

    %% --- Crop back to original size ---
    I_crop = I_filt(1:M, 1:N);

    %% --- Clip and convert ---
    I_out = uint8(max(0, min(255, I_crop)));

end
