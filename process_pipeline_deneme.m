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

    % --- TAHMİNİ DEĞERLER (Blind/No-Reference) ---
    % Salt-Pepper oranını ölç (tahmini)
    sp_ratio_estimated = measure_salt_pepper_ratio(I_degraded);
    
    % Laplacian-MAD yöntemiyle Gaussian noise std tahmini
    gaussian_std_estimated = estimate_gaussian_noise(I_degraded);
    
    % --- GERÇEK DEĞERLER (Original ile karşılaştırma) ---
    % Gerçek Salt-Pepper: Original'dan farklı olan ve 0/255 olan pikseller
    diff_mask = (O ~= D);  % Değişen pikseller
    sp_mask = diff_mask & (D == 0 | D == 255);  % Değişen ve 0/255 olan
    sp_ratio_true = sum(sp_mask(:)) / numel(D);
    
    % Gerçek Gaussian Noise: SP olmayan piksellerdeki farkın std'si
    non_sp_mask = ~(D == 0 | D == 255);  % SP olmayan pikseller
    noise_diff = D(non_sp_mask) - O(non_sp_mask);
    gaussian_std_true = std(noise_diff);
    
    MSE0  = mean((O(:) - D(:)).^2);
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
    sp_threshold = 0.01;  % Salt-pepper oranı bu değerin altına inene kadar devam et
    max_iterations = 10;  % Maksimum iterasyon sayısı (sonsuz döngüyü önlemek için)
    
    I_step1 = I_degraded;
    sp_ratio_current = sp_ratio_estimated;
    iteration = 0;
    
    fprintf("\n--- STEP 1: Iterative Adaptive Median Filter ---\n");
    fprintf("Target S&P Threshold: %.4f (%%%.2f)\n", sp_threshold, sp_threshold*100);
    fprintf("Initial S&P Ratio: %.5f (%%%.2f)\n\n", sp_ratio_current, sp_ratio_current*100);
    
    while sp_ratio_current > sp_threshold && iteration < max_iterations
        iteration = iteration + 1;
        
        % Adaptive median filter uygula
        I_step1 = adaptive_median_filtering(I_step1, 5);
        
        % Yeni salt-pepper oranını ölç
        sp_ratio_current = measure_salt_pepper_ratio(I_step1);
        
        % İlerlemeyi yazdır
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


    %% ============================================================
    % STEP 2 — ITERATIVE NLM (Until Gaussian noise threshold)
    %% ============================================================
    gaussian_threshold = 0.1;  % Gaussian noise bu değerin altına inene kadar devam et
    max_nlm_iterations = 5;    % Maksimum NLM iterasyonu
    
    I_step2 = I_step1;
    gaussian_current = estimate_gaussian_noise(I_step2);
    nlm_iteration = 0;
    
    fprintf("\n--- STEP 2: Iterative NLM (Gaussian Noise Reduction) ---\n");
    fprintf("Target Gaussian Threshold: %.2f\n", gaussian_threshold);
    fprintf("Initial Gaussian Noise (Est.): %.3f\n\n", gaussian_current);
    
    % Grid parameters for NLM (genişletilmiş)
    degree_list = [5 10 15 20 30];  % DegreeOfSmoothing
    sw_list     = [11 21];                     % SearchWindowSize (odd)
    cw_list     = [3 5 7];                        % ComparisonWindowSize (odd)
    
    while gaussian_current > gaussian_threshold && nlm_iteration < max_nlm_iterations
        nlm_iteration = nlm_iteration + 1;
        
        % İterasyon öncesi mevcut durum
        fprintf("  [NLM Iteration %d] - Current Noise: %.3f\n", nlm_iteration, gaussian_current);
        fprintf("  Grid Search başlatılıyor...\n");
        
        % Her iterasyonda en iyi NLM parametrelerini bul (en düşük noise)
        best_noise = inf;  % En düşük noise'u arıyoruz
        best_desc  = "";
        best_candidate = I_step2;
        
        for deg = degree_list
            for sw = sw_list
                for cw = cw_list
                    % Comparison window must be <= search window
                    if cw > sw
                        continue;
                    end
                    
                    candidate = imnlmfilt(I_step2, ...
                        "DegreeOfSmoothing", deg, ...
                        "SearchWindowSize", sw, ...
                        "ComparisonWindowSize", cw);
                    
                    % Gaussian noise tahmini ile scoring
                    noise_est = estimate_gaussian_noise(candidate);
                    
                    fprintf("    deg=%d SW=%d CW=%d -> Noise=%.3f\n", deg, sw, cw, noise_est);
                    
                    if noise_est < best_noise
                        best_noise = noise_est;
                        best_desc  = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                        best_candidate = candidate;
                    end
                end
            end
        end
        
        % Early stopping: Noise düşmeyecekse NLM uygulama
        if best_noise >= gaussian_current
            fprintf("\n    >>> No improvement found (Best: %.3f >= Current: %.3f)\n", ...
                best_noise, gaussian_current);
            fprintf("    >>> Early stopping - NLM iteration terminated.\n\n");
            break;
        end
        
        % Noise düşecekse uygula
        fprintf("\n    >>> Best: %s | Noise: %.3f -> %.3f (Improvement: %.3f)\n\n", ...
            best_desc, gaussian_current, best_noise, gaussian_current - best_noise);
        
        I_step2 = best_candidate;
        gaussian_current = best_noise;
    end
    
    % Sonuç raporu
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


%% ============================================================
% STEP 3 — REPEAT 5x: GRID SEARCH GAUSSIAN FREQ SHARPEN (NO-REFERENCE SCORE)
%% ============================================================
fprintf("\n---- STEP 3: Grid Search (No-Reference Score) x5 ----\n");

% Grid
k_list      = [0.3 0.6 0.9 1.2 1.5 1.8 2.2];
cutoff_list = [0.04 0.06 0.08 0.10 0.12];

lambda = 0.9;   % penalty strength (0.6–1.2 typical)
nRounds = 3;

I_step3 = I_step2;

best_desc3_each  = strings(nRounds,1);
best_score3_each = zeros(nRounds,1);

for r = 1:nRounds
    fprintf("\n  [Round %d/%d]\n", r, nRounds);

    % Baseline metrics from current input
    [edge0, noise0] = edge_noise_metrics(I_step3);

    best_score3 = -inf;
    best_desc3  = "";
    best_img    = I_step3;

    for k = k_list
        for c = cutoff_list
            candidate = freq_highboost_gauss(I_step3, k, c);

            % Compute no-reference score relative to current input
            [edge1, noise1] = edge_noise_metrics(candidate);

            edge_norm  = edge1  / (edge0  + 1e-12);
            noise_norm = noise1 / (noise0 + 1e-12);

            score = edge_norm - lambda * noise_norm;

            fprintf("    k=%.2f c=%.3f | edgeN=%.3f noiseN=%.3f => score=%.4f\n", ...
                k, c, edge_norm, noise_norm, score);

            if score > best_score3
                best_score3 = score;
                best_desc3  = sprintf("k=%.2f,c=%.3f,lambda=%.2f", k, c, lambda);
                best_img    = candidate;
            end
        end
    end

    % update for next round
    I_step3 = best_img;

    best_desc3_each(r)  = string(best_desc3);
    best_score3_each(r) = best_score3;

    fprintf("  >>> Round %d winner: %s | Score=%.4f\n", r, best_desc3, best_score3);
end

fprintf("\n*** STEP 3 FINAL (after %d rounds) ***\n", nRounds);
disp(table((1:nRounds)', best_desc3_each, best_score3_each, ...
    'VariableNames', {'Round','BestParams','BestScore'}));

show_compare_images(I_original, I_step2, I_step3, ...
    sprintf("Step 3: Best Freq Sharpen (NR) x%d", nRounds));
%% ============================================================
% GRID SEARCH: imgaussfilt parameters (maximize PSNR)
%% ============================================================
fprintf("\n---- Grid Search: imgaussfilt (maximize PSNR) ----\n");

% Keep a copy of the input to this step
I_in = I_step3;

sigma_list = [0.2 0.3 0.4 0.5 0.6 0.8 1.0];
fs_list    = [3 5 7 9];  % must be odd

best_psnr = -inf;
best_desc = "";
I_best    = I_in;

for sigma = sigma_list
    for fs = fs_list

        candidate = imgaussfilt(I_in, sigma, ...
            "FilterSize", fs, ...
            "Padding", "symmetric");

        tmp = compare_original_restored(I_original, candidate);

        fprintf("  sigma=%.2f FS=%d -> PSNR=%.4f dB (SSIM=%.4f, MSE=%.4f)\n", ...
            sigma, fs, tmp.PSNR, tmp.SSIM, tmp.MSE);

        if tmp.PSNR > best_psnr
            best_psnr = tmp.PSNR;
            best_desc = sprintf("sigma=%.2f,FS=%d", sigma, fs);
            I_best    = candidate;
        end
    end
end

I_step3 = I_best;

Results3 = compare_original_restored(I_original, I_step3);

    %% ============================================================
    % SUMMARY FOR THIS IMAGE
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 Best NLM (Grid Search)", ...
             "STEP3 Edge-Masked Sharpen"];

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