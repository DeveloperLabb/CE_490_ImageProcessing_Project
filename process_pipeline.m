clc; clear all; close all;

%% ============================================================
% GLOBAL STORAGE FOR ALL IMAGES (Final Step Metrics)
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
    
    best_score = -inf;
    best_params = [0 0];
    
    fprintf("\n---- GRID SEARCH (Bilateral Filter Optimization) ----\n");
    
    for sc = sigmaColor_list
        for ss = sigmaSpatial_list
            candidate = imbilatfilt(I_step1, sc, ss);
            tmp = compare_original_restored(I_original, candidate);
            score = tmp.SSIM + (tmp.PSNR / 50);
    
            fprintf("sigmaColor=%d, sigmaSpatial=%d --> Score=%.4f\n", ...
                sc, ss, score);
    
            if score > best_score
                best_score = score;
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
% STEP 3 — Quantile Laplacian + Soft Sobel + Edge-Masked Sharpen
%% ============================================================

lambda = 0.4;
sobel_weight = 0.5;

fprintf("\n---- STEP 3: Laplacian + Masked Soft Sobel Sharpening ----\n");

%% -------------------------
% 3A — Laplacian
%% -------------------------
I_lap = double(laplacian_filter(I_step2, "lap8"));

show_compare_images(I_original, I_step2, uint8(mat2gray(I_lap)*255), ...
    "Step 3A: Raw Laplacian Edges");


%% -------------------------
% 3B — Threshold Laplacian
%% -------------------------
q = quantile(I_lap(:), 0.25);
I_lap_thresh = I_lap;
I_lap_thresh(abs(I_lap_thresh) < abs(q)) = 0;

show_compare_images(I_original, I_step2, uint8(mat2gray(I_lap_thresh)*255), ...
    "Step 3B: Thresholded Laplacian");


%% -------------------------
% 3C — Soft Sobel
%% -------------------------
[Gx, Gy] = imgradientxy(I_step2);
sobel_mag = abs(Gx) + abs(Gy);
sobel_mag = sobel_mag / max(sobel_mag(:));
sobel_soft = sobel_weight * sobel_mag;

show_compare_images(I_original, I_step2, uint8(sobel_soft*255), ...
    "Step 3C: Soft Sobel");


%% -------------------------
% 3D — Edge Mask
%% -------------------------
edge_strength = abs(I_lap_thresh);
edge_strength = edge_strength / max(edge_strength(:)+1e-8);

edge_mask = max(edge_strength, sobel_soft);
edge_mask = imgaussfilt(edge_mask, 1.2);

show_compare_images(I_original, I_step2, uint8(edge_mask*255), ...
    "Step 3D: Edge Mask");



%% ============================================================
% 3F — FINAL SHARPEN USING ONLY EDGE MASK (NO STEP 3E)
%% ============================================================

% Normalize Laplacian detail
lap_detail_norm = I_lap_thresh / (max(abs(I_lap_thresh(:))) + 1e-8);

% Edge-masked boost
detail_boost = lambda * lap_detail_norm .* edge_mask * 100;

% Apply sharpening
I_step3 = double(I_step2) + detail_boost;
I_step3 = uint8(max(0, min(255, I_step3)));

Results3 = compare_original_restored(I_original, I_step3);

show_compare_images(I_original, I_step2, I_step3, ...
    "Step 3F: Edge-Masked Laplacian Sharpening");




%% ============================================================
% SUMMARY
%% ============================================================
Steps = ["STEP0 Degraded", ...
         "STEP1 Adaptive Median", ...
         "STEP2 Bilateral (Optimized)", ...
         "STEP3 Laplacian + Edge Mask Sharpen"];

MSE_values  = [MSE0,  Results1.MSE,  Results2.MSE,  Results3.MSE];
PSNR_values = [PSNR0, Results1.PSNR, Results2.PSNR, Results3.PSNR];
SSIM_values = [SSIM0, Results1.SSIM, Results2.SSIM, Results3.SSIM];

SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
    'VariableNames', {'Step','MSE','PSNR','SSIM'});

fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
disp(SummaryTable);

All_MSE  = [All_MSE;  Results3.MSE];
All_PSNR = [All_PSNR; Results3.PSNR];
All_SSIM = [All_SSIM; Results3.SSIM];

end % LOOP END




%% ============================================================
% GLOBAL AVERAGES
%% ============================================================
fprintf("\n=============================================\n");
fprintf("     GLOBAL AVERAGE METRICS ACROSS ALL IMAGES\n");
fprintf("=============================================\n");

fprintf("Global Average MSE  = %.4f\n", mean(All_MSE));
fprintf("Global Average PSNR = %.4f dB\n", mean(All_PSNR));
fprintf("Global Average SSIM = %.4f\n", mean(All_SSIM));
fprintf("=============================================\n\n");
