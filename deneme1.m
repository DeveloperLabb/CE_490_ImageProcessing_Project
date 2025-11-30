%% ============================================================
    % STEP 0 — NOISE ANALYSIS
    %% ============================================================
    O = double(I_original);
    D = double(I_degraded);

    num_salt   = sum(D(:) == 255);
    num_pepper = sum(D(:) == 0);
    P_sp = (num_salt + num_pepper) / numel(D);

    Gaussian_std = std((D(:) - O(:)));

    MSE0  = mean((O(:) - D(:)).^2);
    PSNR0 = 10*log10(255^2 / MSE0);
    SSIM0 = ssim(I_degraded, I_original);

    fprintf("\n--- STEP 0 (Original vs Degraded Metrics) ---\n");
    fprintf("Salt-Pepper Ratio: %.5f\n", P_sp);
    fprintf("Gaussian std: %.3f\n", Gaussian_std);
    fprintf("MSE0  = %.4f\n", MSE0);
    fprintf("PSNR0 = %.4f dB\n", PSNR0);
    fprintf("SSIM0 = %.4f\n\n", SSIM0);


    %% ============================================================
    % STEP 1 — ADAPTIVE MEDIAN FILTER
    %% ============================================================
    I_step1 = adaptive_median_filtering(I_degraded, 5);
    Results1 = compare_original_restored(I_original, I_step1);


    %% ============================================================
    % STEP 2 — GAUSSIAN SMOOTHING
    %% ============================================================
    I_step2 = gaussian_smoothing(I_step1, 3, 0.66);
    Results2 = compare_original_restored(I_original, I_step2);


    %% ============================================================
    % STEP 2.5 — SOBEL EDGE DETECTION
    %% ============================================================
    [Gx, Gy, Gmag] = sobel_filter(I_step2);


    %% ============================================================
    % STEP 3 — HYBRID GAUSSIAN + BUTTERWORTH SHARPENING
    %% ============================================================
    I_step3 = fft_hybrid_sharpen(I_step2, 1.2, 25, 1.0, 2, 40);
    Results3 = compare_original_restored(I_original, I_step3);


    %% ============================================================
    % STEP 4 — BILATERAL FILTER
    %% ============================================================
    I_step4 = imbilatfilt(I_step3, 200, 15);
    Results4 = compare_original_restored(I_original, I_step4);


    %% ============================================================
    % STEP 5 — SOBEL EDGE FUSION
    %% ============================================================
    I_step5 = uint8(max(0, min(255, double(I_step4) + 0.15 * double(Gmag))));
    Results5 = compare_original_restored(I_original, I_step5);