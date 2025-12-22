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
    
    Io = im2double(I_original);
    Ir = im2double(I_degraded);

    MSE0   = immse(Ir, Io);
    PSNR0 = 10*log10(255^2 / (MSE0 + 1e-12));
    SSIM0 = ssim(I_degraded, I_original);

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
    gaussian_threshold  = 0.1;
    max_nlm_iterations  = 5;

    I_step2 = I_step1;
    gaussian_current = estimate_gaussian_noise(I_step2);
    nlm_iteration = 0;

    fprintf("\n--- STEP 2: Iterative NLM (Gaussian Noise Reduction) ---\n");
    fprintf("Target Gaussian Threshold: %.2f\n", gaussian_threshold);
    fprintf("Initial Gaussian Noise (Est.): %.3f\n\n", gaussian_current);

    degree_list = [5 10 15 20 30];
    sw_list     = [11 21];
    cw_list     = [3 5 7];

    while gaussian_current > gaussian_threshold && nlm_iteration < max_nlm_iterations
        nlm_iteration = nlm_iteration + 1;

        fprintf("  [NLM Iteration %d] - Current Noise: %.3f\n", nlm_iteration, gaussian_current);
        fprintf("  Grid Search başlatılıyor...\n");

        best_noise     = inf;
        best_desc      = "";
        best_candidate = I_step2;

        for deg = degree_list
            for sw = sw_list
                for cw = cw_list
                    if cw > sw, continue; end

                    candidate = imnlmfilt(I_step2, ...
                        "DegreeOfSmoothing", deg, ...
                        "SearchWindowSize", sw, ...
                        "ComparisonWindowSize", cw);

                    noise_est = estimate_gaussian_noise(candidate);

                    fprintf("    deg=%d SW=%d CW=%d -> Noise=%.3f\n", deg, sw, cw, noise_est);

                    if noise_est < best_noise
                        best_noise     = noise_est;
                        best_desc      = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                        best_candidate = candidate;
                    end
                end
            end
        end

        if best_noise >= gaussian_current
            fprintf("\n    >>> No improvement found (Best: %.3f >= Current: %.3f)\n", ...
                best_noise, gaussian_current);
            fprintf("    >>> Early stopping - NLM iteration terminated.\n\n");
            break;
        end

        fprintf("\n    >>> Best: %s | Noise: %.3f -> %.3f (Improvement: %.3f)\n\n", ...
            best_desc, gaussian_current, best_noise, gaussian_current - best_noise);

        I_step2 = best_candidate;
        gaussian_current = best_noise;
    end

    if gaussian_current <= gaussian_threshold
        fprintf("  >> Gaussian threshold (%.2f) reached after %d iterations!\n", ...
            gaussian_threshold, nlm_iteration);
    elseif nlm_iteration >= max_nlm_iterations
        fprintf("  >> Max NLM iterations (%d) reached. Final Gaussian: %.3f\n", ...
            max_nlm_iterations, gaussian_current);
    else
        fprintf("  >> Early stopped at iteration %d. Final Gaussian: %.3f\n", ...
            nlm_iteration, gaussian_current);
    end

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
    fprintf("\n--- STEP 3: Iterative Freq Sharpen (Edge > Noise Gain) ---\n");

    max_sharpen_iterations = 5;
    I_step3 = I_step2;

    [edge_current, noise_current] = edge_noise_metrics(I_step3);
    sharpen_iteration = 0;

    fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_current, noise_current);
    fprintf("Rule: Continue while edge_gain > noise_gain\n\n");

    k_list      = [0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.8 2.0];
    cutoff_list = [0.03 0.05 0.07 0.09 0.11 0.13];

    while sharpen_iteration < max_sharpen_iterations
        sharpen_iteration = sharpen_iteration + 1;

        fprintf("  [Sharpen Iteration %d] - Edge: %.6f | Noise: %.6f\n", ...
            sharpen_iteration, edge_current, noise_current);
        fprintf("  Grid Search başlatılıyor...\n");

        [edge0, noise0] = edge_noise_metrics(I_step3);

        best_ratio      = inf;
        best_edge_gain  = 0;
        best_noise_gain = 0;
        best_desc       = "";
        best_candidate  = I_step3;

        for k = k_list
            for c = cutoff_list
                candidate = freq_highboost_gauss(I_step3, k, c);

                [edge1, noise1] = edge_noise_metrics(candidate);

                edge_norm  = edge1  / (edge0  + 1e-12);
                noise_norm = noise1 / (noise0 + 1e-12);

                edge_gain  = edge_norm  - 1.0;
                noise_gain = noise_norm - 1.0;

                if edge_gain > 0
                    ratio = noise_gain / edge_gain;
                else
                    ratio = inf;
                end

                fprintf("    k=%.2f c=%.3f | edge_gain=%.4f noise_gain=%.4f | ratio=%.4f\n", ...
                    k, c, edge_gain, noise_gain, ratio);

                if ratio < best_ratio
                    best_ratio      = ratio;
                    best_edge_gain  = edge_gain;
                    best_noise_gain = noise_gain;
                    best_desc       = sprintf("k=%.2f,c=%.3f", k, c);
                    best_candidate  = candidate;
                end
            end
        end

        ratio_threshold = 3.0;
        if best_ratio >= ratio_threshold
            fprintf("\n    >>> Stop: best ratio = %.4f >= %.1f threshold\n", ...
                best_ratio, ratio_threshold);
            fprintf("    >>> Noise increasing too fast. Early stopping.\n\n");
            break;
        end

        [edge_new, noise_new] = edge_noise_metrics(best_candidate);

        noise_threshold = 0.03;
        if noise_new > noise_threshold
            fprintf("\n    >>> Stop: noise = %.6f > %.4f threshold\n", ...
                noise_new, noise_threshold);
            fprintf("    >>> Noise too high. Not applying this iteration.\n\n");
            break;
        end

        fprintf("\n    >>> Best: %s\n", best_desc);
        fprintf("    >>> Edge gain: %.4f | Noise gain: %.4f | Ratio: %.4f\n", ...
            best_edge_gain, best_noise_gain, best_ratio);
        fprintf("    >>> Edge: %.6f -> %.6f | Noise: %.6f -> %.6f\n\n", ...
            edge_current, edge_new, noise_current, noise_new);

        I_step3 = best_candidate;
        edge_current = edge_new;
        noise_current = noise_new;
    end

    if sharpen_iteration >= max_sharpen_iterations
        fprintf("  >> Max sharpen iterations (%d) reached.\n", max_sharpen_iterations);
    else
        fprintf("  >> Early stopped at iteration %d.\n", sharpen_iteration);
    end

    fprintf("\n*** STEP 3 FINAL ***\n");
    fprintf("Total Iterations: %d\n", sharpen_iteration);
    fprintf("Final Edge: %.6f | Noise: %.6f\n\n", edge_current, noise_current);

    show_compare_images(I_original, I_step2, I_step3, ...
        sprintf("Step 3: Iterative Sharpen (%d iters)", sharpen_iteration));

    % Save STEP 3 output
    save_step_image(OUT_DIR, name, 4, "step3_sharpen", I_step3);
    

    %% ============================================================
    % STEP 4 — EDGE-MASKED LAPLACIAN ENHANCEMENT
    %% ============================================================
    fprintf("\n--- STEP 4: Edge-Masked Laplacian Enhancement ---\n");
    
    laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    I_step3_double = im2double(I_step3);
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

    I_step2_double = im2double(I_step2);
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
    fprintf("\n--- STEP 5: Iterative Gaussian Smoothing ---\n");

    max_step5_iterations = 5;
    I_step5 = I_final;

    [edge_initial, ~] = edge_noise_metrics(I_step5);
    [edge_current, noise_current] = edge_noise_metrics(I_step5);
    step5_iteration = 0;
    total_edge_loss = 0;

    fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_initial, noise_current);
    fprintf("Rule: noise_red/edge_loss max, stop if total edge loss > 5%%\n\n");

    sigma_list = [0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0];
    fs_list    = [3 5 7 9 11 13];

    while step5_iteration < max_step5_iterations
        step5_iteration = step5_iteration + 1;

        fprintf("  [Iteration %d] - Edge: %.6f | Noise: %.6f | Total Edge Loss: %.2f%%\n", ...
            step5_iteration, edge_current, noise_current, total_edge_loss*100);
        fprintf("  Grid Search başlatılıyor...\n");

        [edge0, noise0] = edge_noise_metrics(I_step5);

        best_ratio = -inf;
        best_noise = inf;
        best_desc  = "";
        best_candidate = I_step5;
        found_valid = false;

        for sigma = sigma_list
            for fs = fs_list
                candidate = imgaussfilt(I_step5, sigma, ...
                    "FilterSize", fs, ...
                    "Padding", "symmetric");

                [edge1, noise1] = edge_noise_metrics(candidate);

                edge_loss       = (edge0 - edge1) / (edge0 + 1e-12);
                noise_reduction = (noise0 - noise1) / (noise0 + 1e-12);

                potential_total_loss = (edge_initial - edge1) / (edge_initial + 1e-12);

                if edge_loss > 0.001
                    ratio = noise_reduction / edge_loss;
                else
                    ratio = noise_reduction * 100;
                end

                fprintf("    sigma=%.2f FS=%d | edge_loss=%.4f noise_red=%.4f ratio=%.2f total=%.2f%%\n", ...
                    sigma, fs, edge_loss, noise_reduction, ratio, potential_total_loss*100);

                if potential_total_loss < 0.1 && noise_reduction > 0 && ratio > best_ratio
                    best_ratio = ratio;
                    best_noise = noise1;
                    best_desc  = sprintf("sigma=%.2f,FS=%d", sigma, fs);
                    best_candidate = candidate;
                    found_valid = true;
                end
            end
        end

        if ~found_valid || best_noise >= noise_current
            fprintf("\n    >>> Stop: no valid parameter (noise reduction + edge loss < 5%%)\n");
            fprintf("    >>> Early stopping.\n\n");
            break;
        end

        [edge_new, noise_new] = edge_noise_metrics(best_candidate);
        total_edge_loss = (edge_initial - edge_new) / (edge_initial + 1e-12);

        fprintf("\n    >>> Best: %s | Ratio: %.2f\n", best_desc, best_ratio);
        fprintf("    >>> Noise: %.6f -> %.6f\n", noise_current, noise_new);
        fprintf("    >>> Total Edge Loss: %.2f%%\n\n", total_edge_loss*100);

        I_step5 = best_candidate;
        edge_current = edge_new;
        noise_current = noise_new;
    end

    if step5_iteration >= max_step5_iterations
        fprintf("  >> Max iterations (%d) reached.\n", max_step5_iterations);
    else
        fprintf("  >> Early stopped at iteration %d.\n", step5_iteration);
    end

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
    gaussian_threshold6 = 0.08;
    max_nlm_iterations6 = 5;

    I_step6 = I_step5;
    gaussian6_current = estimate_gaussian_noise(I_step6);
    nlm6_iteration = 0;

    fprintf("\n--- STEP 6: Iterative NLM Again (Gaussian Noise Reduction) ---\n");
    fprintf("Target Gaussian Threshold: %.2f\n", gaussian_threshold6);
    fprintf("Initial Gaussian Noise (Est.): %.3f\n\n", gaussian6_current);

    degree_list6 = [5 10 15 20 30];
    sw_list6     = [11 21];
    cw_list6     = [3 5 7];

    while gaussian6_current > gaussian_threshold6 && nlm6_iteration < max_nlm_iterations6
        nlm6_iteration = nlm6_iteration + 1;

        fprintf("  [NLM6 Iteration %d] - Current Noise: %.3f\n", nlm6_iteration, gaussian6_current);
        fprintf("  Grid Search başlatılıyor...\n");

        best_noise6     = inf;
        best_desc6      = "";
        best_candidate6 = I_step6;

        for deg = degree_list6
            for sw = sw_list6
                for cw = cw_list6
                    if cw > sw, continue; end

                    candidate = imnlmfilt(I_step6, ...
                        "DegreeOfSmoothing", deg, ...
                        "SearchWindowSize", sw, ...
                        "ComparisonWindowSize", cw);

                    noise_est = estimate_gaussian_noise(candidate);

                    fprintf("    deg=%d SW=%d CW=%d -> Noise=%.3f\n", deg, sw, cw, noise_est);

                    if noise_est < best_noise6
                        best_noise6     = noise_est;
                        best_desc6      = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                        best_candidate6 = candidate;
                    end
                end
            end
        end

        if best_noise6 >= gaussian6_current
            fprintf("\n    >>> No improvement found (Best: %.3f >= Current: %.3f)\n", ...
                best_noise6, gaussian6_current);
            fprintf("    >>> Early stopping - Step 6 terminated.\n\n");
            break;
        end

        fprintf("\n    >>> Best: %s | Noise: %.3f -> %.3f (Improvement: %.3f)\n\n", ...
            best_desc6, gaussian6_current, best_noise6, gaussian6_current - best_noise6);

        I_step6 = best_candidate6;
        gaussian6_current = best_noise6;
    end

    if gaussian6_current <= gaussian_threshold6
        fprintf("  >> Gaussian threshold (%.2f) reached after %d iterations!\n", ...
            gaussian_threshold6, nlm6_iteration);
    elseif nlm6_iteration >= max_nlm_iterations6
        fprintf("  >> Max NLM iterations (%d) reached. Final Gaussian: %.3f\n", ...
            max_nlm_iterations6, gaussian6_current);
    else
        fprintf("  >> Early stopped at iteration %d. Final Gaussian: %.3f\n", ...
            nlm6_iteration, gaussian6_current);
    end

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
