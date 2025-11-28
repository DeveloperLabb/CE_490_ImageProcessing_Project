function I_out = laplacian_sharpen_quantile(I_in, lambda, sigma)
% LAPLACIAN_SHARPEN_QUANTILE
% 1) Laplacian edges extracted
% 2) Quantile thresholding (1st quantile → zayıf gürültü yok edilir)
% 3) Gaussian smoothing (halo removal)
% 4) Fusion with base image

    %% ============================================================
    % 1. Raw Laplacian (strong edges)
    %% ============================================================
    I_lap = laplacian_filter(I_in, "lap8");    % uses your existing function
    I_lap = double(I_lap);

    %% ============================================================
    % 2. 1st Quantile Thresholding
    %    Removes weak / noisy Laplacian responses
    %% ============================================================
    q = quantile(I_lap(:), 0.25);   % 25st percentile
    
    % Suppress small edges
    I_lap_thresh = I_lap;
    I_lap_thresh(abs(I_lap_thresh) < abs(q)) = 0;

    %% ============================================================
    % 3. Smooth thresholded Laplacian (halo cleanup)
    %% ============================================================
    I_lap_smooth = imgaussfilt(I_lap_thresh, sigma);

    %% ============================================================
    % 4. Fusion with base image
    %% ============================================================
    I_out = double(I_in) + lambda * I_lap_smooth;

    % Clip to uint8 range
    I_out = uint8(max(0, min(255, I_out)));

end
