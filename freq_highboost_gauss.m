function Iout_u8 = freq_highboost_gauss(Iin_u8, k, cutoff)
% Gaussian frequency-domain high-boost sharpening.
% cutoff: normalized radius (~0.03–0.15 typical)
% k: boost strength

    I = im2double(Iin_u8);
    [H,W] = size(I);

    F = fftshift(fft2(I));

    [u,v] = meshgrid( (-floor(W/2):ceil(W/2)-1)/W, ...
                      (-floor(H/2):ceil(H/2)-1)/H );
    D = sqrt(u.^2 + v.^2);

    Hlp = exp(-(D.^2) / (2*cutoff^2));  % Gaussian low-pass
    Hhp = 1 - Hlp;                      % high-pass

    G  = 1 + k * Hhp;                   % high-boost gain

    Irec = real(ifft2(ifftshift(F .* G)));
    Irec = min(max(Irec,0),1);

    Iout_u8 = im2uint8(Irec);
end
