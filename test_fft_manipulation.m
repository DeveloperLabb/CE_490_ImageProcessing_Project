clc; clear all; close all;

%% ============================================================
% GLOBAL STORAGE FOR FINAL STEP METRICS (STEP 4: ASE + Sharpen)
%% ============================================================
All_MSE  = [];
All_PSNR = [];
All_SSIM = [];


%% ============================================================
% IMAGE LIST
%% ============================================================
image_names = {
    "boat", ...
    "baboon", ...
    "barbara", ...
    "peppers", ...
    "cameraman" ...
};


%% ============================================================
% MAIN LOOP FOR ALL IMAGES
%% ============================================================
for idx = 1:length(image_names)

    name = image_names{idx};

    fprintf("\n=============================================\n");
    fprintf("        PROCESSING IMAGE: %s\n", upper(name));
    fprintf("=============================================\n");

    %% LOAD IMAGES
    I_original = uint8(imread("project_images/clean/original_" + name + ".png"));
    I_degraded = uint8(imread("project_images/degraded/degraded_" + name + ".png"));


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

    show_compare_images(I_original, I_degraded, I_step1, ...
        "Step 1: Adaptive Median Filter");



    %% ============================================================
    % STEP 2 — GRID SEARCH OPTIMIZED BILATERAL FILTER
    %% ============================================================
    sigmaColor_list   = [40, 80, 120, 160];
    sigmaSpatial_list = [4, 6, 8, 10];

    best_score  = -inf;
    best_params = [0 0];

    fprintf("\n---- GRID SEARCH (Bilateral Filter Optimization) ----\n");

    for sc = sigmaColor_list
        for ss = sigmaSpatial_list

            candidate = imbilatfilt(I_step1, sc, ss);
            tmp = compare_original_restored(I_original, candidate);

            % Combined metric: SSIM + PSNR/50
            score = tmp.SSIM + (tmp.PSNR / 50);

            fprintf("sigmaColor=%d, sigmaSpatial=%d --> Score=%.4f\n", ...
                sc, ss, score);

            if score > best_score
                best_score  = score;
                best_params = [sc, ss];
            end
        end
    end

    fprintf("\n*** OPTIMAL BILATERAL PARAMETERS FOUND ***\n");
    fprintf("sigmaColor   = %d\n", best_params(1));
    fprintf("sigmaSpatial = %d\n", best_params(2));
    fprintf("BEST SCORE   = %.4f\n\n", best_score);

    I_step2 = imbilatfilt(I_step1, best_params(1), best_params(2));

    Results2 = compare_original_restored(I_original, I_step2);

    show_compare_images(I_original, I_step1, I_step2, ...
        sprintf("Step 2: Optimized Bilateral (C=%d, S=%d)", ...
            best_params(1), best_params(2)));



    %% ============================================================
    % STEP 3 — AUTOREGRESSIVE SPECTRAL EXTRAPOLATION (ASE)
    % Automatic AR Order Selection via AIC
    %% ============================================================
    
    fprintf("\n---- STEP 3: Adaptive ASE with Automatic AR Order ----\n");
    
    % FFT
    F = fft2(double(I_step2));
    Fshift = fftshift(F);
    
    % Adaptive radius based on low-frequency energy
    [M, N] = size(F);
    [xx, yy] = meshgrid(1:N, 1:M);
    cx = N/2;  cy = M/2;
    
    radius_min = 0.10 * min(M,N);
    radius_max = 0.30 * min(M,N);
    
    best_radius = radius_min;
    best_energy = -inf;
    
    for r = round(linspace(radius_min, radius_max, 6))
        mask = (xx - cx).^2 + (yy - cy).^2 <= r^2;
        energy = sum(abs(Fshift(mask)),'all');
        if energy > best_energy
            best_energy = energy;
            best_radius = r;
        end
    end
    
    low_mask = (xx - cx).^2 + (yy - cy).^2 <= best_radius^2;
    
    LowPart   = Fshift .* low_mask;
    known     = LowPart(low_mask);
    known_mag = abs(known);
    
    %% === AUTOMATIC AR ORDER SELECTION (AIC) ===
    orders   = 4:30;  
    best_AIC = inf;
    best_order = 12;
    
    for p = orders
        [a_tmp, noise_pow_tmp] = aryule(known_mag, p);
    
        AIC = length(known_mag) * log(noise_pow_tmp) + 2*p;
    
        if AIC < best_AIC
            best_AIC  = AIC;
            best_order = p;
        end
    end
    
    fprintf("Selected AR order (AIC-based): %d\n", best_order);
    
    % Final AR model using best order
    [a, noise_pow] = aryule(known_mag, best_order);
    
    pred_len = length(F(:)) - length(known_mag);
    
    predicted_mag = filter(-a(2:end), 1, zeros(pred_len,1));
    
    % Normalize predicted magnitude to LF dynamic range
    predicted_mag = predicted_mag / max(predicted_mag(:)+1e-12) * max(known_mag);
    
    HF_indices = find(~low_mask);
    F_new      = Fshift;
    F_new(HF_indices) = predicted_mag .* exp(1i * angle(Fshift(HF_indices)));
    
    % inverse FFT → ASE deblurred image
    I_ase = real(ifft2(ifftshift(F_new)));
    I_ase = uint8(max(0, min(255, I_ase)));
    
    Results3 = compare_original_restored(I_original, I_ase);
    
    show_compare_images(I_original, I_step2, I_ase, ...
        sprintf("Step 3: ASE Deblur (Adaptive Mask, AR=%d)", best_order));



    %% ============================================================
    % STEP 4 — LAPLACIAN + SOFT SOBEL EDGE-MASK SHARPENING
    %% ============================================================

    fprintf("\n---- STEP 4: Laplacian + Soft Sobel Edge-Masked Sharpen ----\n");

    % Çalışma görüntüsü: ASE çıktısı
    I_base = I_ase;

    % Parametreler (biraz daha güvenli ama görünür keskinlik)
    lambda       = 0.35;   % sharpen strength
    sobel_weight = 0.5;    % soft Sobel contribution
    boost_scale  = 60;     % 100 çok agresifti, 50–70 arası iyi

    %% 4A — Raw Laplacian
    I_lap = double(laplacian_filter(I_base, "lap8"));

    show_compare_images(I_original, I_base, uint8(mat2gray(I_lap)*255), ...
        "Step 4A: Raw Laplacian Edges");

    %% 4B — Thresholded Laplacian (weaker edges removed)
    q = quantile(I_lap(:), 0.25);
    I_lap_thresh = I_lap;
    I_lap_thresh(abs(I_lap_thresh) < abs(q)) = 0;

    show_compare_images(I_original, I_base, uint8(mat2gray(I_lap_thresh)*255), ...
        "Step 4B: Thresholded Laplacian");

    %% 4C — Soft Sobel
    [Gx, Gy] = imgradientxy(I_base);
    sobel_mag = abs(Gx) + abs(Gy);
    sobel_mag = sobel_mag / max(sobel_mag(:) + 1e-8);
    sobel_soft = sobel_weight * sobel_mag;

    show_compare_images(I_original, I_base, uint8(sobel_soft*255), ...
        "Step 4C: Soft Sobel");

    %% 4D — Edge Mask (Laplacian + Sobel birleştirme)
    edge_strength = abs(I_lap_thresh);
    edge_strength = edge_strength / max(edge_strength(:) + 1e-8);

    edge_mask = max(edge_strength, sobel_soft);
    edge_mask = imgaussfilt(edge_mask, 1.2);  % smooth mask

    show_compare_images(I_original, I_base, uint8(edge_mask*255), ...
        "Step 4D: Edge Mask");

    %% 4E — Final Sharpen (Edge-masked Laplacian detail)
    lap_detail_norm = I_lap_thresh / (max(abs(I_lap_thresh(:))) + 1e-8);

    detail_boost = lambda * lap_detail_norm .* edge_mask * boost_scale;

    I_step4 = double(I_base) + detail_boost;
    I_step4 = uint8(max(0, min(255, I_step4)));

    Results4 = compare_original_restored(I_original, I_step4);

    show_compare_images(I_original, I_base, I_step4, ...
        "Step 4E: Final Edge-Masked Laplacian Sharpen");


    %% ============================================================
    % SUMMARY FOR THIS IMAGE
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 Bilateral (Optimized)", ...
             "STEP3 ASE Deblur", ...
             "STEP4 ASE + Edge-Masked Sharpen"];

    MSE_values  = [MSE0,  Results1.MSE,  Results2.MSE,  Results3.MSE, Results4.MSE];
    PSNR_values = [PSNR0, Results1.PSNR, Results2.PSNR, Results3.PSNR, Results4.PSNR];
    SSIM_values = [SSIM0, Results1.SSIM, Results2.SSIM, Results3.SSIM, Results4.SSIM];

    SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
        'VariableNames', {'Step','MSE','PSNR','SSIM'});

    fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
    disp(SummaryTable);

    %% Store final (Step 4) metrics as "final pipeline quality"
    All_MSE  = [All_MSE;  Results4.MSE];
    All_PSNR = [All_PSNR; Results4.PSNR];
    All_SSIM = [All_SSIM; Results4.SSIM];

end % LOOP END



%% ============================================================
% GLOBAL AVERAGES (FINAL STEP = ASE + SHARPEN)
%% ============================================================
fprintf("\n===============================================================\n");
fprintf("   GLOBAL AVERAGE METRICS (Final Step = ASE + Edge Sharpening)\n");
fprintf("===============================================================\n");

fprintf("Global Average MSE   = %.4f\n", mean(All_MSE));
fprintf("Global Average PSNR  = %.4f dB\n", mean(All_PSNR));
fprintf("Global Average SSIM  = %.4f\n", mean(All_SSIM));
fprintf("===============================================================\n\n");
