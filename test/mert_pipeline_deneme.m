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
    % STEP 1 — ADAPTIVE MEDIAN FILTER (YOUR EXISTING FUNCTION)
    %% ============================================================
    I_step1 = adaptive_median_filtering(I_degraded, 5);

    Results1 = compare_original_restored(I_original, I_step1);

    show_compare_images(I_original, I_degraded, I_step1, ...
        "Step 1: Adaptive Median Filter");


    %% ============================================================
    % STEP 2 — GRID SEARCH: ONLY NON-LOCAL MEANS (imnlmfilt)
    %% ============================================================
    fprintf("\n---- STEP 2: GRID SEARCH (NLM only: imnlmfilt) ----\n");

    best_score  = -inf;
    best_desc   = "";
    I_step2     = I_step1;

    % Grid parameters (feel free to expand, but this is a good start)
    degree_list = [5 10 15 20 30];
    sw_list     = [11 21];        % SearchWindowSize (odd)
    cw_list     = [3 5 7];        % ComparisonWindowSize (odd)

    for deg = degree_list
        for sw = sw_list
            for cw = cw_list

                % Comparison window must be <= search window (and both odd)
                if cw > sw
                    continue;
                end

                candidate = imnlmfilt(I_step1, ...
                    "DegreeOfSmoothing", deg, ...
                    "SearchWindowSize", sw, ...
                    "ComparisonWindowSize", cw);

                tmp = compare_original_restored(I_original, candidate);

                % Same scoring logic you used before:
                score = tmp.SSIM + (tmp.PSNR / 50);

                fprintf("  NLM deg=%d SW=%d CW=%d --> Score=%.4f (PSNR=%.3f, SSIM=%.4f)\n", ...
                    deg, sw, cw, score, tmp.PSNR, tmp.SSIM);

                if score > best_score
                    best_score = score;
                    best_desc  = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                    I_step2    = candidate;
                end
            end
        end
    end

    Results2 = compare_original_restored(I_original, I_step2);

    fprintf("\n*** STEP 2 WINNER (NLM) ***\n");
    fprintf("Params   : %s\n", best_desc);
    fprintf("BestScore: %.4f\n\n", best_score);

    show_compare_images(I_original, I_step1, I_step2, ...
        sprintf("Step 2: Best NLM (%s)", best_desc));


    %% ============================================================
    % STEP 3–7 — YOUR NEW PIPELINE:
    % 3) Laplacian of STEP1
    % 4) Threshold Laplacian (abs) to suppress noise
    % 5) Erode + Dilate mask
    % 6) Mask * Laplacian (signed)
    % 7) Add to STEP2
    %% ============================================================
    fprintf("\n---- STEP 3-7: Laplacian Mask Reinjection Pipeline ----\n");

    % ---------- Tunable params ----------
    lap_alpha  = 0.2;   % fspecial('laplacian', alpha)
    thr_mult   = 1.00;  % >1 stricter mask, <1 more edges
    se_radius  = 1;     % morphology radius
    edge_gain  = 1.0;   % how much masked Laplacian is added to NLM output
    do_close   = true;  % optional: close after open (reduces small holes)
    % -----------------------------------

    % (3) Laplacian of STEP1 (signed)
    I1d = double(I_step1);
    hL  = fspecial('laplacian', lap_alpha);
    L   = imfilter(I1d, hL, 'replicate', 'conv');  % signed Laplacian

    % (4) Threshold abs(L) -> mask
    absL = abs(L);
    absL_norm = mat2gray(absL);              % normalize to [0,1] for Otsu
    T = graythresh(absL_norm) * thr_mult;    % Otsu threshold (scaled)
    M = absL_norm > T;

    % (5) Erode + Dilate (opening) to remove minimal noise, optional close
    se = strel('disk', se_radius, 0);
    M5 = imdilate(imerode(M, se), se);       % open = erode then dilate
    if do_close
        M5 = imerode(imdilate(M5, se), se);  % close = dilate then erode
    end

    % (6) Multiply mask * Laplacian (keep signed)
    D6 = double(M5) .* L;

    % (7) Sum STEP2 (NLM) + masked Laplacian
    I2d = double(I_step2);
    I7  = I2d + edge_gain * D6;

    % clamp to [0,255] and cast back to uint8
    I7 = uint8(min(max(I7, 0), 255));

    I_step3 = I7;  % final output of the new pipeline
    Results3 = compare_original_restored(I_original, I_step3);

    show_compare_images(I_original, I_step2, I_step3, ...
        sprintf("Step 3-7: NLM + Masked Laplacian (thr=%.3f, gain=%.2f)", T, edge_gain));


    %% ============================================================
    % SUMMARY FOR THIS IMAGE
    %% ============================================================
    Steps = ["STEP0 Degraded", ...
             "STEP1 Adaptive Median", ...
             "STEP2 Best NLM (Grid Search)", ...
             "STEP3-7 NLM + Masked Laplacian"];

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
