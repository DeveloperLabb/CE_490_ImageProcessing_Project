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
    % STEP 1 — ADAPTIVE MEDIAN FILTER (UNTIL SALT & PEPPER RATIO)
    %% ============================================================
    fprintf("\n---- STEP 1: Adaptive Median (Salt & Pepper Ratio Based) ----\n");

    % ====== THRESHOLD ======
    TARGET_SP_RATIO = 0.01;   % örn: %1'in altına düşünce dur
    MAX_PASSES      = 10;     % güvenlik için
    KERNEL_SIZE     = 5;

    I_step1 = I_degraded;

    for pass = 1:MAX_PASSES

        % one filtering pass
        I_step1 = adaptive_median_filtering(I_step1, KERNEL_SIZE);

        % ====== SALT & PEPPER RATIO ======
        % 0 ve 255 olan piksel oranı
        sp_pixels = (I_step1 == 0) | (I_step1 == 255);
        sp_ratio  = sum(sp_pixels(:)) / numel(I_step1);

        % quality metrics
        Results1 = compare_original_restored(I_original, I_step1);

        % report every pass
        fprintf("Pass %d: SP_Ratio=%.4f | MSE=%.2f | PSNR=%.2f dB | SSIM=%.4f\n", ...
            pass, sp_ratio, Results1.MSE, Results1.PSNR, Results1.SSIM);

        % stopping condition
        if sp_ratio <= TARGET_SP_RATIO
            fprintf("STOP: Salt & Pepper ratio threshold reached at pass %d.\n", pass);
            break;
        end

        if pass == MAX_PASSES
            fprintf("STOP: MAX_PASSES=%d reached (SP_Ratio=%.4f).\n", MAX_PASSES, sp_ratio);
        end
    end

    show_compare_images(I_original, I_degraded, I_step1, ...
        "Step 1: Adaptive Median (Salt & Pepper Ratio Controlled)");
    
    %% ============================================================
    % STEP 2_B — ADAPTIVE BILATERAL FILTER (SSIM-DRIVEN + ROLLBACK)
    %% ============================================================
    fprintf("\n---- STEP 2: Adaptive Bilateral Filtering (SSIM-driven, rollback) ----\n");
    
    % ====== TARGETS ======
    TARGET_GAUSS_STD = 12;
    MAX_PASSES       = 6;
    SIGMA_SPATIAL    = 5;
    
    % Initial sigmaColor (noise seviyesine bağlı)
    sigmaColor = max(30, Gaussian_std * 1.5);
    
    I_step2b = I_step1;
    
    best_I_step2 = I_step2b;   % en iyi görüntüyü tut
    best_SSIM    = -inf;
    
    for pass = 1:MAX_PASSES
    
        % Apply bilateral filter
        I_candidate = imbilatfilt(I_step2b, sigmaColor, SIGMA_SPATIAL);
    
        % ===== Residual Gaussian estimation (impulse masked) =====
        O = double(I_original);
        R = double(I_candidate);
        mask = (R ~= 0) & (R ~= 255);
        residual_gauss_std = std(R(mask) - O(mask));
    
        % Metrics
        Results2 = compare_original_restored(I_original, I_candidate);
    
        % Report
        fprintf("Pass %d: sigmaColor=%.1f | Residual σ=%.2f | MSE=%.2f | PSNR=%.2f dB | SSIM=%.4f\n", ...
            pass, sigmaColor, residual_gauss_std, ...
            Results2.MSE, Results2.PSNR, Results2.SSIM);
    
        % ===== BEST STATE CHECK =====
        if Results2.SSIM > best_SSIM
            best_SSIM    = Results2.SSIM;
            best_I_step2 = I_candidate;
        else
            % SSIM düştü → rollback
            fprintf("ROLLBACK: SSIM decreased. Reverting to best result (SSIM=%.4f).\n", best_SSIM);
            break;
        end
    
        % Gaussian noise yeterince bastırıldıysa dur
        if residual_gauss_std <= TARGET_GAUSS_STD
            fprintf("STOP: Gaussian noise threshold reached.\n");
            break;
        end
    
        % Prepare for next iteration
        I_step2b   = I_candidate;
        sigmaColor = sigmaColor * 0.75;
    end
    
    % Use best result
    I_step2b = best_I_step2;
    
    show_compare_images(I_original, I_step1, I_step2b, ...
    "Step 2_B: Adaptive Bilateral (SSIM-driven + Rollback)");
    
    %% ============================================================
    % STEP 3 — LAPLACIAN EDGE EXTRACTION (AFTER DENOISING)
    %% ============================================================
    fprintf("\n---- STEP 3: Laplacian Edge Extraction (Post-Bilateral) ----\n");
    
    % Laplacian edge extraction on denoised image
    I_step3_edges = double(laplacian_filter(I_step2b, "lap8"));
    
    % For visualization only (no enhancement)
    I_step3_vis = uint8(mat2gray(abs(I_step3_edges)) * 255);
    
    fprintf("Edge extraction complete. Range: [%.2f, %.2f]\n", ...
        min(I_step3_edges(:)), max(I_step3_edges(:)));
    
    show_compare_images(I_original, I_step2b, I_step3_vis, ...
        "Step 3: Laplacian Edges (After Adaptive Bilateral)");

    %% ============================================================
    % STEP 3_B — SMOOTHED LAPLACIAN ADDITION (EDGE ENHANCEMENT)
    %% ============================================================
    fprintf("\n---- STEP 3_B: Smoothed Laplacian Enhancement ----\n");
    
    % --- Take SIGNED Laplacian (directional edges preserved)
    L = I_step3_edges;
    
    % --- Smooth Laplacian response to suppress noise amplification
    % Gaussian smoothing on edge map
    sigma_L = 1.0;   % small sigma → edge-preserving
    L_smooth = imgaussfilt(L, sigma_L);
    
    % --- Normalize Laplacian response (CRITICAL)
    L_smooth = L_smooth / (max(abs(L_smooth(:))) + 1e-8);
    
    % --- Adaptive scaling factor (safe range)
    lambda = 0.4;   % edge enhancement strength
    
    % --- Add Laplacian back to denoised image
    I_step3_enhanced = double(I_step2b) + lambda * L_smooth;
    
    % --- Clip to valid intensity range
    I_step3_enhanced = uint8(max(0, min(255, I_step3_enhanced)));
    
    % --- Metrics (optional but recommended)
    Results3 = compare_original_restored(I_original, I_step3_enhanced);
    
    fprintf("After Laplacian enhancement: MSE=%.2f | PSNR=%.2f dB | SSIM=%.4f\n", ...
        Results3.MSE, Results3.PSNR, Results3.SSIM);
    
    show_compare_images(I_original, I_step2b, I_step3_enhanced, ...
        "Step 3_B: Smoothed Laplacian Enhancement");
end % LOOP END

