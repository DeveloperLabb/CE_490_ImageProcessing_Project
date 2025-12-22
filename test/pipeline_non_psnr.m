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

    %% LOAD IMAGES (GRAYSCALE)
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
    PSNR0 = 10*log10(255^2 / (MSE0 + 1e-12));
    SSIM0 = ssim(I_degraded, I_original);

    fprintf("\n--- STEP 0 (Original vs Degraded Metrics) ---\n");
    fprintf("Salt-Pepper Ratio: %.5f\n", P_sp);
    fprintf("Gaussian std: %.3f\n", Gaussian_std);
    fprintf("MSE0  = %.4f\n", MSE0);
    fprintf("PSNR0 = %.4f dB\n", PSNR0);
    fprintf("SSIM0 = %.4f\n\n", SSIM0);


    %% ============================================================
    % STEP 1 — ITERATIVE ADAPTIVE MEDIAN FILTER (Until S&P Normalized)
    %% ============================================================
    fprintf("\n---- STEP 1: Iterative Adaptive Median ----\n");
    
    % Calculate target: ratio of 254/1 pixels (normal background)
    target_ratio = (sum(D(:) == 254) + sum(D(:) == 1)) / numel(D);
    fprintf("  Target ratio (254/1): %.5f\n", target_ratio);
    fprintf("  Initial S&P ratio (255/0): %.5f\n\n", P_sp);
    
    I_step1 = I_degraded;
    maxIter = 5;
    med_iters = 0;
    
    for i = 1:maxIter
        % Apply adaptive median
        I_step1 = adaptive_median_filtering(I_step1, 5);
        
        % Calculate current S&P ratio
        curr_sp = (sum(double(I_step1(:)) == 255) + sum(double(I_step1(:)) == 0)) / numel(I_step1);
        
        med_iters = i;
        fprintf("  Iter %d: S&P ratio = %.5f\n", i, curr_sp);
        
        % Stop if S&P ratio approaches target (within 2x)
        if curr_sp <= target_ratio * 2
            fprintf("  [OK] S&P normalized after %d iterations\n", i);
            break;
        end
    end

    % Reference metrics for reporting only
    Results1 = compare_original_restored(I_original, I_step1);
    fprintf("STEP 1 Result: PSNR=%.2f dB, SSIM=%.4f (%d iters)\n", Results1.PSNR, Results1.SSIM, med_iters);

    show_compare_images(I_original, I_degraded, I_step1, ...
        sprintf("%s | Step 1: AdaptiveMedian (%dx)", upper(name), med_iters));


    %% ============================================================
    % STEP 2 — GRID SEARCH: NLM (NO-REFERENCE: edge_noise_metrics)
    %% ============================================================
    fprintf("\n---- STEP 2: GRID SEARCH (NLM - No-Reference) ----\n");

    % Baseline edge/noise metrics
    [edge0, noise0] = edge_noise_metrics(I_step1);
    fprintf("  Baseline: Edge=%.4f, Noise=%.4f\n\n", edge0, noise0);

    best_score  = -inf;
    best_desc   = "";
    I_step2     = I_step1;

    % Grid parameters
    degree_list = [5 10 15 20 30];
    sw_list     = [11 21];
    cw_list     = [3 5 7];

    for deg = degree_list
        for sw = sw_list
            for cw = cw_list
                if cw > sw, continue; end

                candidate = imnlmfilt(I_step1, ...
                    "DegreeOfSmoothing", deg, ...
                    "SearchWindowSize", sw, ...
                    "ComparisonWindowSize", cw);

                % No-reference scoring
                [edge1, noise1] = edge_noise_metrics(candidate);
                
                edge_ratio  = edge1 / (edge0 + 1e-12);   % want ~1
                noise_ratio = noise1 / (noise0 + 1e-12); % want < 1
                
                % Reward noise reduction, penalize edge loss
                noise_reward = max(0, 1 - noise_ratio);
                edge_penalty = max(0, 1 - edge_ratio);
                
                score = noise_reward - 0.5 * edge_penalty;

                fprintf("  NLM deg=%2d SW=%2d CW=%d --> EdgeR=%.3f NoiseR=%.3f | Score=%.4f\n", ...
                    deg, sw, cw, edge_ratio, noise_ratio, score);

                if score > best_score
                    best_score = score;
                    best_desc  = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                    I_step2    = candidate;
                end
            end
        end
    end
    % Reference metrics for reporting only
    Results2 = compare_original_restored(I_original, I_step2);

    fprintf("\n*** STEP 2 WINNER (NLM - No-Reference) ***\n");
    fprintf("Params: %s | Score=%.4f\n", best_desc, best_score);
    fprintf("STEP 2 Result: PSNR=%.2f dB, SSIM=%.4f\n\n", Results2.PSNR, Results2.SSIM);

    show_compare_images(I_original, I_step1, I_step2, ...
        sprintf("%s | Step 2: NLM (1x) | %s", upper(name), best_desc));


%% ============================================================
% STEP 3 — ADAPTIVE FREQ SHARPEN (Convergence-Based)
%% ============================================================
fprintf("\n---- STEP 3: Adaptive Freq Sharpen (Auto-Convergence) ----\n");

% Grid
k_list      = [0.3 0.6 0.9 1.2 1.5 1.8 2.2];
cutoff_list = [0.04 0.06 0.08 0.10 0.12];

lambda = 0.9;

% Adaptive parameters (max 3 rounds)
maxRounds      = 3;
minImprovement = 0.005;   
noImproveTol   = 2;      

I_step3 = I_step2;

% Dynamic storage
best_desc3_all  = {};
best_score3_all = [];
prev_best_score = -inf;
noImproveCount  = 0;
actualRounds    = 0;

for r = 1:maxRounds
    fprintf("\n  [Round %d/%d (max)]\n", r, maxRounds);

    % Baseline metrics from current input
    [edge0, noise0] = edge_noise_metrics(I_step3);

    best_score3 = -inf;
    best_desc3  = "";
    best_img    = I_step3;

    for k = k_list
        for c = cutoff_list
            candidate = freq_highboost_gauss(I_step3, k, c);

            % Compute no-reference score
            [edge1, noise1] = edge_noise_metrics(candidate);

            edge_norm  = edge1  / (edge0  + 1e-12);
            noise_norm = noise1 / (noise0 + 1e-12);

            score = edge_norm - lambda * noise_norm;

            if score > best_score3
                best_score3 = score;
                best_desc3  = sprintf("k=%.2f,c=%.3f", k, c);
                best_img    = candidate;
            end
        end
    end

    % Store results
    best_desc3_all{r}  = best_desc3;
    best_score3_all(r) = best_score3;
    actualRounds = r;

    % Convergence check
    improvement = best_score3 - prev_best_score;
    fprintf("  >>> Round %d: %s | Score=%.4f | Improve=%.4f\n", ...
        r, best_desc3, best_score3, improvement);

    if improvement < minImprovement && r > 1
        noImproveCount = noImproveCount + 1;
        fprintf("      [!] Low improvement (%d/%d)\n", noImproveCount, noImproveTol);
        
        if noImproveCount >= noImproveTol
            fprintf("\n  [STOP] Converged after %d rounds\n", r);
            break;
        end
    else
        noImproveCount = 0;
    end

    I_step3 = best_img;
    prev_best_score = best_score3;
end

% Reference metrics for reporting only
Results3 = compare_original_restored(I_original, I_step3);

fprintf("\n*** STEP 3 FINAL (after %d adaptive rounds) ***\n", actualRounds);
resultTable = table((1:actualRounds)', string(best_desc3_all(1:actualRounds))', ...
    best_score3_all(1:actualRounds)', 'VariableNames', {'Round','BestParams','BestScore'});
disp(resultTable);
fprintf("STEP 3 Result: PSNR=%.2f dB, SSIM=%.4f\n", Results3.PSNR, Results3.SSIM);

show_compare_images(I_original, I_step2, I_step3, ...
    sprintf("%s | Step 3: FreqSharpen (%dx) | %s", upper(name), actualRounds, best_desc3));

%% ============================================================
% STEP 4 — ITERATIVE GAUSSIAN SMOOTHING (Based on STEP 3 rounds)
%% ============================================================
% Number of iterations = max(1, floor(sharpening_rounds / 3))
nGaussIter = max(1, floor(actualRounds / 3));
fprintf("\n---- STEP 4: Gaussian Smoothing (%dx based on %d sharpen rounds) ----\n", nGaussIter, actualRounds);

I_step4 = I_step3;

sigma_list = [0.2 0.3 0.4 0.5 0.6 0.8 1.0];
fs_list    = [3 5 7];

for iter = 1:nGaussIter
    fprintf("\n  [Gauss Iter %d/%d]\n", iter, nGaussIter);
    
    % Baseline for this iteration
    [edge0, noise0] = edge_noise_metrics(I_step4);
    
    best_score = -inf;
    best_sigma = 0.3;
    best_fs = 3;
    
    % Grid search for best params
    for sigma = sigma_list
        for fs = fs_list
            candidate = imgaussfilt(I_step4, sigma, ...
                "FilterSize", fs, ...
                "Padding", "symmetric");

            [edge1, noise1] = edge_noise_metrics(candidate);
            
            edge_ratio  = edge1 / (edge0 + 1e-12);
            noise_ratio = noise1 / (noise0 + 1e-12);
            
            noise_reward = max(0, 1 - noise_ratio);
            edge_penalty = max(0, 1 - edge_ratio);
            
            score = noise_reward - 0.15 * edge_penalty;

            if score > best_score
                best_score = score;
                best_sigma = sigma;
                best_fs = fs;
            end
        end
    end
    
    % Apply best params
    I_step4 = imgaussfilt(I_step4, best_sigma, ...
        "FilterSize", best_fs, ...
        "Padding", "symmetric");
    
    fprintf("  >>> Iter %d: sigma=%.2f, FS=%d | Score=%.4f\n", iter, best_sigma, best_fs, best_score);
end

best_desc = sprintf("sigma=%.2f,FS=%d", best_sigma, best_fs);

% Reference metrics for reporting only
Results4 = compare_original_restored(I_original, I_step4);
fprintf("\n*** STEP 4 DONE (%d iterations) ***\n", nGaussIter);
fprintf("STEP 4 Result: PSNR=%.2f dB, SSIM=%.4f\n", Results4.PSNR, Results4.SSIM);

step4_title = sprintf("%s | Step 4: Gauss (%dx) | %s", upper(name), nGaussIter, best_desc);
show_compare_images(I_original, I_step3, I_step4, step4_title);

    %% ============================================================
    % SUMMARY FOR THIS IMAGE
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 NLM Smoothing", ...
             "STEP3 Freq Sharpening", ...
             "STEP4 Gaussian Smooth"];

    MSE_values  = [MSE0,  Results1.MSE,  Results2.MSE,  Results3.MSE,  Results4.MSE];
    PSNR_values = [PSNR0, Results1.PSNR, Results2.PSNR, Results3.PSNR, Results4.PSNR];
    SSIM_values = [SSIM0, Results1.SSIM, Results2.SSIM, Results3.SSIM, Results4.SSIM];

    SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
        'VariableNames', {'Step','MSE','PSNR','SSIM'});

    fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
    disp(SummaryTable);

    % Store final metrics for global averages
    All_MSE  = [All_MSE;  Results4.MSE];
    All_PSNR = [All_PSNR; Results4.PSNR];
    All_SSIM = [All_SSIM; Results4.SSIM];

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