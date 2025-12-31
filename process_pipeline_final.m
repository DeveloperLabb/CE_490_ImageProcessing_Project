clc; clear all; close all;

%% ============================================================
% GLOBAL STORAGE FOR ALL IMAGES (Final Step Metrics)
%% ============================================================
All_MSE  = [];
All_PSNR = [];
All_SSIM = [];

%% ============================================================
% OUTPUTS FOLDER
%% ============================================================
OUT_DIR = "outputs";   % you said you created this folder already

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

    %% LOAD IMAGES (GRAYSCALE)
    I_original = uint8(imread("project_images/clean/original_" + name + ".png"));
    I_degraded = uint8(imread("project_images/degraded/degraded_" + name + ".png"));

    % Save ORIGINAL + DEGRADED immediately
    save_step_image(OUT_DIR, name, 0, "original", I_original);
    save_step_image(OUT_DIR, name, 1, "degraded", I_degraded);

    %% ============================================================
    % STEP 0 — NOISE ANALYSIS
    %% ============================================================
    O = double(I_original);
    D = double(I_degraded);

    % --- TAHMİNİ DEĞERLER (Blind/No-Reference) ---
    sp_ratio_estimated     = measure_salt_pepper_ratio(I_degraded);
    gaussian_std_estimated = estimate_gaussian_noise(I_degraded);

    % --- GERÇEK DEĞERLER (Original ile karşılaştırma) ---
    diff_mask     = (O ~= D);
    sp_mask       = diff_mask & (D == 0 | D == 255);
    sp_ratio_true = sum(sp_mask(:)) / numel(D);

    non_sp_mask       = ~(D == 0 | D == 255);
    noise_diff        = D(non_sp_mask) - O(non_sp_mask);
    gaussian_std_true = std(noise_diff);
    
    Io = double(I_original) / 255;
    Ir = double(I_degraded) / 255;

    MSE0  = immse(Ir, Io);
    PSNR0 = psnr(Ir, Io, 1);
    SSIM0 = ssim(Ir, Io, "DynamicRange", 1);

    fprintf("\n--- STEP 0 (Original vs Degraded Metrics) ---\n");
    fprintf("Salt-Pepper Ratio:\n");
    fprintf("  - Estimated (0/255 count): %.5f (%%%.2f)\n", sp_ratio_estimated, sp_ratio_estimated*100);
    fprintf("  - True (from Original):    %.5f (%%%.2f)\n", sp_ratio_true, sp_ratio_true*100);
    fprintf("Gaussian Noise Std:\n");
    fprintf("  - Estimated (Laplacian-MAD): %.3f\n", gaussian_std_estimated);
    fprintf("  - True (from Original):      %.3f\n", gaussian_std_true);
    fprintf("MSE0  = %.4f\n", MSE0);
    fprintf("PSNR0 = %.4f dB\n", PSNR0);
    fprintf("SSIM0 = %.4f\n\n", SSIM0);

    %% ============================================================
    % STEP 1 — ITERATIVE ADAPTIVE MEDIAN FILTER (Until threshold)
    %% ============================================================
    sp_threshold    = 0.01;
    max_iterations  = 10;

    I_step1 = I_degraded;
    sp_ratio_current = sp_ratio_estimated;
    iteration = 0;

    fprintf("\n--- STEP 1: Iterative Adaptive Median Filter ---\n");
    fprintf("Target S&P Threshold: %.4f (%%%.2f)\n", sp_threshold, sp_threshold*100);
    fprintf("Initial S&P Ratio: %.5f (%%%.2f)\n\n", sp_ratio_current, sp_ratio_current*100);

    while sp_ratio_current > sp_threshold && iteration < max_iterations
        iteration = iteration + 1;

        I_step1 = adaptive_median_filtering(I_step1, 5);
        sp_ratio_current = measure_salt_pepper_ratio(I_step1);

        fprintf("  Iteration %d: S&P Ratio = %.5f (%%%.2f)\n", ...
            iteration, sp_ratio_current, sp_ratio_current*100);
    end

    if sp_ratio_current <= sp_threshold
        fprintf("\n  >> Threshold reached after %d iterations!\n", iteration);
    else
        fprintf("\n  >> Max iterations (%d) reached. Final S&P Ratio: %.5f\n", ...
            max_iterations, sp_ratio_current);
    end

    Results1 = compare_original_restored(I_original, I_step1);

    show_compare_images(I_original, I_degraded, I_step1, ...
        "Step 1: Adaptive Median Filter");

    % Save STEP 1 output
    save_step_image(OUT_DIR, name, 2, "step1_adaptive_median", I_step1);

    %% ============================================================
    % STEP 2 — ITERATIVE NLM (Until Gaussian noise threshold)
    %% ============================================================
    degree_list = [5 10 15 20 30];
    sw_list     = [11 21];
    cw_list     = [3 5 7];

    [I_step2, gaussian_current, nlm_iteration] = iterative_nlm_filter(...
        I_step1, 0.1, 5, degree_list, sw_list, cw_list, "STEP 2");

    Results2 = compare_original_restored(I_original, I_step2);

    fprintf("\n*** STEP 2 FINAL RESULT ***\n");
    fprintf("Total Iterations: %d\n", nlm_iteration);
    fprintf("Final Gaussian Noise: %.3f\n", gaussian_current);
    fprintf("PSNR: %.4f dB | SSIM: %.4f\n\n", Results2.PSNR, Results2.SSIM);

    show_compare_images(I_original, I_step1, I_step2, ...
        sprintf("Step 2: Iterative NLM (%d iters)", nlm_iteration));

    % Save STEP 2 output
    save_step_image(OUT_DIR, name, 3, "step2_nlm", I_step2);

    %% ============================================================
    % STEP 3 — ITERATIVE FREQ SHARPEN (Edge/Noise Ratio with Early Stopping)
    %% ============================================================
    k_list      = [0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.8 2.0];
    cutoff_list = [0.03 0.05 0.07 0.09 0.11 0.13];

    [I_step3, edge_current, noise_current, sharpen_iteration] = iterative_freq_sharpen(...
        I_step2, 5, k_list, cutoff_list, 3.0, 0.03);

    show_compare_images(I_original, I_step2, I_step3, ...
        sprintf("Step 3: Iterative Sharpen (%d iters)", sharpen_iteration));

    % Save STEP 3 output
    save_step_image(OUT_DIR, name, 4, "step3_sharpen", I_step3);
    

    %% ============================================================
    % STEP 4 — EDGE-MASKED LAPLACIAN ENHANCEMENT
    %% ============================================================
    fprintf("\n--- STEP 4: Edge-Masked Laplacian Enhancement ---\n");
    
    laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    I_step3_double = double(I_step3) / 255;
    laplace_output = imfilter(I_step3_double, laplacian_kernel, 'symmetric');

    laplace_vis = mat2gray(laplace_output);   % scales min..max -> 0..1
    laplace_vis_u8 = im2uint8(laplace_vis);
    save_step_image(OUT_DIR, name, 45, "laplace_output", laplace_vis_u8);

    
    enhanced_edges = double(laplace_output);

    [~, noise_sigma] = edge_noise_metrics(I_step3_double);
    noise_var = noise_sigma^2;

    lap_smooth_w = wiener2(enhanced_edges, [3 3], noise_var);

    meanKernel = fspecial('average', [3 3]);
    lap_smooth = imfilter(lap_smooth_w, meanKernel, 'symmetric');

    I_step2_double = double(I_step2) / 255;
    I_final = I_step2_double + lap_smooth;

    I_final = max(0, min(1, I_final));
    I_final = im2uint8(I_final);

    Results4 = compare_original_restored(I_original, I_final);

    fprintf("\n*** STEP 4 FINAL RESULT ***\n");
    fprintf("PSNR: %.4f dB | SSIM: %.4f\n\n", Results4.PSNR, Results4.SSIM);

    figure('Name', sprintf('Step 4 Process - %s', name));
    subplot(2,3,1); imshow(I_step2); title('Step 2 (NLM)');
    subplot(2,3,2); imshow(laplace_output, []); title('Laplacian of Step 3');
    subplot(2,3,3); imshow(enhanced_edges, []); title('Laplacian');
    subplot(2,3,4); imshow(I_final); title('Final Result');
    subplot(2,3,5); imshow(I_original); title('Original');
    sgtitle(sprintf('Step 4: Edge-Masked Laplacian - %s', upper(name)));

    show_compare_images(I_original, I_step3, I_final, ...
        "Step 4: Edge-Masked Laplacian Enhancement");

    % Save STEP 4 output
    save_step_image(OUT_DIR, name, 5, "step4_laplacian", I_final);

    %% ============================================================
    % STEP 5 — ITERATIVE GAUSSIAN SMOOTHING (Edge Preserve + Noise Reduce)
    %% ============================================================
    sigma_list = [0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0];
    fs_list    = [3 5 7 9 11 13];

    [I_step5, edge_current, noise_current, step5_iteration] = iterative_gaussian_smooth(...
        I_final, 5, sigma_list, fs_list);

    Results5 = compare_original_restored(I_original, I_step5);

    fprintf("\n*** STEP 5 FINAL RESULT ***\n");
    fprintf("Total Iterations: %d\n", step5_iteration);
    fprintf("Final Edge: %.6f | Noise: %.6f\n", edge_current, noise_current);
    fprintf("PSNR: %.4f dB | SSIM: %.4f\n\n", Results5.PSNR, Results5.SSIM);

    show_compare_images(I_original, I_final, I_step5, ...
        sprintf("Step 5: Iterative Gaussian (%d iters)", step5_iteration));

    % Save STEP 5 output
    save_step_image(OUT_DIR, name, 6, "step5_gaussian", I_step5);

    %% ============================================================
    % STEP 6 — ITERATIVE NLM AGAIN (Same as Step 2)
    %% ============================================================
    degree_list6 = [5 10 15 20 30];
    sw_list6     = [11 21];
    cw_list6     = [3 5 7];

    [I_step6, gaussian6_current, nlm6_iteration] = iterative_nlm_filter(...
        I_step5, 0.08, 5, degree_list6, sw_list6, cw_list6, "STEP 6");

    Results6 = compare_original_restored(I_original, I_step6);

    fprintf("\n*** STEP 6 FINAL RESULT ***\n");
    fprintf("Total Iterations: %d\n", nlm6_iteration);
    fprintf("Final Gaussian Noise: %.3f\n", gaussian6_current);
    fprintf("PSNR: %.4f dB | SSIM: %.4f\n\n", Results6.PSNR, Results6.SSIM);

    show_compare_images(I_original, I_step5, I_step6, ...
        sprintf("Step 6: Iterative NLM Again (%d iters)", nlm6_iteration));

    % Save STEP 6 output (FINAL)
    save_step_image(OUT_DIR, name, 7, "step6_nlm_again_final", I_step6);

    %% ============================================================
    % METRIC SUMMARY
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 NLM", ...
             "STEP4 Edge-Masked Laplacian", ...
             "STEP5 Final Gaussian", ...
             "STEP6 Brief NLM Again"];

    MSE_values  = [MSE0,  Results1.MSE,  Results2.MSE,  Results4.MSE,  Results5.MSE,  Results6.MSE];
    PSNR_values = [PSNR0, Results1.PSNR, Results2.PSNR, Results4.PSNR, Results5.PSNR, Results6.PSNR];
    SSIM_values = [SSIM0, Results1.SSIM, Results2.SSIM, Results4.SSIM, Results5.SSIM, Results6.SSIM];

    SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
        'VariableNames', {'Step','MSE','PSNR','SSIM'});

    fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
    disp(SummaryTable);

    % Final outputs should come from Step 6 now
    All_MSE  = [All_MSE;  Results6.MSE];
    All_PSNR = [All_PSNR; Results6.PSNR];
    All_SSIM = [All_SSIM; Results6.SSIM];

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
