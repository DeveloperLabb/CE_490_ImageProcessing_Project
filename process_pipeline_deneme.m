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
% STEP 3 — ITERATIVE FREQ SHARPEN (Edge/Noise Ratio with Early Stopping)
%% ============================================================
fprintf("\n--- STEP 3: Iterative Freq Sharpen (Edge > Noise Gain) ---\n");

max_sharpen_iterations = 5;   % Maksimum iterasyon

I_step3 = I_step2;

% Başlangıç değerlerini hesapla
[edge_current, noise_current] = edge_noise_metrics(I_step3);
sharpen_iteration = 0;

fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_current, noise_current);
fprintf("Rule: Continue while edge_gain > noise_gain\n\n");

% Grid parameters
k_list      = [0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.8 2.0];
cutoff_list = [0.03 0.05 0.07 0.09 0.11 0.13];

while sharpen_iteration < max_sharpen_iterations
    sharpen_iteration = sharpen_iteration + 1;
    
    % İterasyon öncesi mevcut durum
    fprintf("  [Sharpen Iteration %d] - Edge: %.6f | Noise: %.6f\n", ...
        sharpen_iteration, edge_current, noise_current);
    fprintf("  Grid Search başlatılıyor...\n");
    
    % Baseline metrics (normalize için)
    [edge0, noise0] = edge_noise_metrics(I_step3);
    
    % Her iterasyonda en iyi parametreleri bul
    % En iyi = noise_gain / edge_gain oranı en düşük olan
    best_ratio = inf;  % noise_gain / edge_gain (düşük = iyi)
    best_edge_gain = 0;
    best_noise_gain = 0;
    best_desc  = "";
    best_candidate = I_step3;
    
    for k = k_list
        for c = cutoff_list
            candidate = freq_highboost_gauss(I_step3, k, c);
            
            % edge_noise_metrics ile scoring
            [edge1, noise1] = edge_noise_metrics(candidate);
            
            % Normalize edilmiş oranlar
            edge_norm  = edge1 / (edge0 + 1e-12);
            noise_norm = noise1 / (noise0 + 1e-12);
            
            % Gain hesapla
            edge_gain = edge_norm - 1.0;   % Edge ne kadar arttı
            noise_gain = noise_norm - 1.0; % Noise ne kadar arttı
            
            % Oran: noise_gain / edge_gain (düşük = daha iyi)
            % edge_gain <= 0 ise çok kötü, büyük oran ver
            if edge_gain > 0
                ratio = noise_gain / edge_gain;
            else
                ratio = inf;  % Edge artmadıysa skip
            end
            
            fprintf("    k=%.2f c=%.3f | edge_gain=%.4f noise_gain=%.4f | ratio=%.4f\n", ...
                k, c, edge_gain, noise_gain, ratio);
            
            % En düşük ratio olanı seç
            if ratio < best_ratio
                best_ratio = ratio;
                best_edge_gain = edge_gain;
                best_noise_gain = noise_gain;
                best_desc = sprintf("k=%.2f,c=%.3f", k, c);
                best_candidate = candidate;
            end
        end
    end
    
    % Early stopping 1: ratio >= 3 ise dur
    ratio_threshold = 3.0;
    
    if best_ratio >= ratio_threshold
        fprintf("\n    >>> Stop: best ratio = %.4f >= %.1f threshold\n", ...
            best_ratio, ratio_threshold);
        fprintf("    >>> Noise increasing too fast. Early stopping.\n\n");
        break;
    end
    
    % En iyi adayın noise değerini kontrol et
    [edge_new, noise_new] = edge_noise_metrics(best_candidate);
    
    % Early stopping 2: Noise 0.03'ün üstüne çıkarsa dur
    noise_threshold = 0.03;
    
    if noise_new > noise_threshold
        fprintf("\n    >>> Stop: noise = %.6f > %.4f threshold\n", ...
            noise_new, noise_threshold);
        fprintf("    >>> Noise too high. Not applying this iteration.\n\n");
        break;
    end
    
    % Uygula
    fprintf("\n    >>> Best: %s\n", best_desc);
    fprintf("    >>> Edge gain: %.4f | Noise gain: %.4f | Ratio: %.4f\n", ...
        best_edge_gain, best_noise_gain, best_ratio);
    fprintf("    >>> Edge: %.6f -> %.6f | Noise: %.6f -> %.6f\n\n", ...
        edge_current, edge_new, noise_current, noise_new);
    
    I_step3 = best_candidate;
    edge_current = edge_new;
    noise_current = noise_new;
end

% Sonuç raporu
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

% Canny Edge Detection Comparison
figure('Name', sprintf('Canny Edge - %s', name));
subplot(1,3,1);
imshow(edge(I_original, 'canny'));
title('Original - Canny Edge');

subplot(1,3,2);
imshow(edge(I_step2, 'canny'));
title('Step 2 (NLM) - Canny Edge');

subplot(1,3,3);
imshow(edge(I_step3, 'canny'));
title('Step 3 (Sharpen) - Canny Edge');

sgtitle(sprintf('Canny Edge Comparison - %s', upper(name)));

%% ============================================================
% STEP 4 — EDGE-MASKED LAPLACIAN ENHANCEMENT
%% ============================================================
fprintf("\n--- STEP 4: Edge-Masked Laplacian Enhancement ---\n");

% 1. I_step2 (NLM) → Canny edge → Erode → Dilate → edge mask
canny_edge = edge(I_step2, 'canny');

% Morphological operations
%se = strel('square', 3);  % Yapısal eleman
%edge_eroded = imerode(canny_edge, se);
edge_mask = canny_edge;

fprintf("Canny Edge: %d edge pixels\n", sum(canny_edge(:)));
fprintf("After Erode->Dilate: %d edge pixels\n", sum(edge_mask(:)));

% 2. I_step3 → Laplacian filter (sadece çıktı, enhance yok)
laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
I_step3_double = im2double(I_step3);
laplace_output = imfilter(I_step3_double, laplacian_kernel, 'symmetric');

% 3. Laplacian çıktısı × Edge mask
edge_mask_double = double(edge_mask);
enhanced_edges = double(laplace_output);

% Robust Gaussian noise variance estimate (MAD)
[dummy , noise_sigma] = edge_noise_metrics(I_step3_double);
noise_var   = noise_sigma^2;

% Wiener filter (küçük pencere = edge koruma)
lap_smooth_w = wiener2(enhanced_edges, [3 3], noise_var);

% 2️⃣ Light mean filter (VERY SMALL kernel)
meanKernel = fspecial('average', [3 3]);   % 3x3 max!
lap_smooth = imfilter(lap_smooth_w, ...
                      meanKernel, ...
                      'symmetric');

% 4. Sonuç + I_step2 = Final
I_step2_double = im2double(I_step2);
I_final = I_step2_double + lap_smooth;

% Clip to valid range [0, 1]
I_final = max(0, min(1, I_final));
I_final = im2uint8(I_final);

Results4 = compare_original_restored(I_original, I_final);

fprintf("\n*** STEP 4 FINAL RESULT ***\n");
fprintf("PSNR: %.4f dB | SSIM: %.4f\n\n", Results4.PSNR, Results4.SSIM);

% Visualization
figure('Name', sprintf('Step 4 Process - %s', name));
subplot(2,3,1);
imshow(I_step2);
title('Step 2 (NLM)');

subplot(2,3,2);
imshow(edge_mask);
title('Edge Mask (Canny→Erode→Dilate)');

subplot(2,3,3);
imshow(laplace_output, []);
title('Laplacian of Step 3');

subplot(2,3,4);
imshow(enhanced_edges, []);
title('Laplacian × Edge Mask');

subplot(2,3,5);
imshow(I_final);
title('Final Result');

subplot(2,3,6);
imshow(I_original);
title('Original');

sgtitle(sprintf('Step 4: Edge-Masked Laplacian - %s', upper(name)));

show_compare_images(I_original, I_step3, I_final, ...
    "Step 4: Edge-Masked Laplacian Enhancement");

%% ============================================================
% STEP 5 — ITERATIVE GAUSSIAN SMOOTHING (Edge Preserve + Noise Reduce)
%% ============================================================
fprintf("\n--- STEP 5: Iterative Gaussian Smoothing ---\n");

max_step5_iterations = 5;
I_step5 = I_final;

% Başlangıç edge/noise değerleri
[edge_initial, noise_initial] = edge_noise_metrics(I_step5);
[edge_current, noise_current] = edge_noise_metrics(I_step5);
step5_iteration = 0;
total_edge_loss = 0;  % Toplam edge kaybı

fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_initial, noise_initial);
fprintf("Rule: noise_red/edge_loss max, stop if total edge loss > 5%%\n\n");

% Grid parameters
sigma_list = [0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0];
fs_list    = [3 5 7 9 11 13];

while step5_iteration < max_step5_iterations
    step5_iteration = step5_iteration + 1;
    
    fprintf("  [Iteration %d] - Edge: %.6f | Noise: %.6f | Total Edge Loss: %.2f%%\n", ...
        step5_iteration, edge_current, noise_current, total_edge_loss*100);
    fprintf("  Grid Search başlatılıyor...\n");
    
    % Baseline
    [edge0, noise0] = edge_noise_metrics(I_step5);
    
    % En iyi: noise_red / edge_loss oranı en yüksek olan
    best_ratio = -inf;
    best_noise = inf;
    best_edge_loss = 0;
    best_desc = "";
    best_candidate = I_step5;
    found_valid = false;
    
    for sigma = sigma_list
        for fs = fs_list
            candidate = imgaussfilt(I_step5, sigma, ...
                "FilterSize", fs, ...
                "Padding", "symmetric");
            
            [edge1, noise1] = edge_noise_metrics(candidate);
            
            % Edge kaybı ve noise azalması
            edge_loss = (edge0 - edge1) / (edge0 + 1e-12);
            noise_reduction = (noise0 - noise1) / (noise0 + 1e-12);
            
            % Bu parametre uygulanırsa total edge loss ne olur?
            potential_total_loss = (edge_initial - edge1) / (edge_initial + 1e-12);
            
            % noise_red / edge_loss oranı
            if edge_loss > 0.001
                ratio = noise_reduction / edge_loss;
            else
                ratio = noise_reduction * 100;
            end
            
            fprintf("    sigma=%.2f FS=%d | edge_loss=%.4f noise_red=%.4f ratio=%.2f total=%.2f%%\n", ...
                sigma, fs, edge_loss, noise_reduction, ratio, potential_total_loss*100);
            
            % Sadece total edge loss %5'in altında ve noise azalıyorsa seç
            if potential_total_loss < 0.1 && noise_reduction > 0 && ratio > best_ratio
                best_ratio = ratio;
                best_noise = noise1;
                best_edge_loss = edge_loss;
                best_edge_new = edge1;
                best_desc = sprintf("sigma=%.2f,FS=%d", sigma, fs);
                best_candidate = candidate;
                found_valid = true;
            end
        end
    end
    
    % Early stopping: %5 altında noise azaltan parametre yoksa dur
    if ~found_valid || best_noise >= noise_current
        fprintf("\n    >>> Stop: no valid parameter (noise reduction + edge loss < 5%%)\n");
        fprintf("    >>> Early stopping.\n\n");
        break;
    end
    
    % Uygula (zaten %5'in altında olanlar seçildi)
    [edge_new, noise_new] = edge_noise_metrics(best_candidate);
    total_edge_loss = (edge_initial - edge_new) / (edge_initial + 1e-12);
    
    fprintf("\n    >>> Best: %s | Ratio: %.2f\n", best_desc, best_ratio);
    fprintf("    >>> Edge loss: %.2f%% | Noise: %.6f -> %.6f\n", ...
        best_edge_loss*100, noise_current, noise_new);
    fprintf("    >>> Total Edge Loss: %.2f%%\n\n", total_edge_loss*100);
    
    I_step5 = best_candidate;
    edge_current = edge_new;
    noise_current = noise_new;
end

% Sonuç
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

    %% ============================================================
    % SUMMARY FOR THIS IMAGE
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 NLM", ...
             "STEP4 Edge-Masked Laplacian", ...
             "STEP5 Final Gaussian"];

    MSE_values  = [MSE0,  Results1.MSE,  Results2.MSE,  Results4.MSE,  Results5.MSE];
    PSNR_values = [PSNR0, Results1.PSNR, Results2.PSNR, Results4.PSNR, Results5.PSNR];
    SSIM_values = [SSIM0, Results1.SSIM, Results2.SSIM, Results4.SSIM, Results5.SSIM];

    SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
        'VariableNames', {'Step','MSE','PSNR','SSIM'});

    fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
    disp(SummaryTable);

    All_MSE  = [All_MSE;  Results5.MSE];
    All_PSNR = [All_PSNR; Results5.PSNR];
    All_SSIM = [All_SSIM; Results5.SSIM];

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