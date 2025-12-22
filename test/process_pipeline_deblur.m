clc; clear; close all;

%% ============================================================
% PIPELINE (GRAYSCALE)
% 0) Baseline metrics + noise analysis
% 1) Adaptive Median Filter
% 2) SA-DCT Blind Deblur + Denoise  (sadct_blind_deblur)
% 3) Non-Local Means (NLM) AFTER SA-DCT (light cleanup)
%% ============================================================

addpath(genpath("C:\Users\husey\OneDrive\Desktop\CE_490\SA_DCT"));

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
    % STEP 0 — NOISE ANALYSIS + BASELINE METRICS
    %% ============================================================
    O01 = im2double(I_original);
    D01 = im2double(I_degraded);

    num_salt   = sum(I_degraded(:) == 255);
    num_pepper = sum(I_degraded(:) == 0);
    P_sp = (num_salt + num_pepper) / numel(I_degraded);

    Gaussian_std = std((D01(:) - O01(:)));  % [0,1] scale

    Results0 = compare_original_restored(I_original, I_degraded);

    fprintf("\n--- STEP 0 (Original vs Degraded Metrics) ---\n");
    fprintf("Salt-Pepper Ratio: %.5f\n", P_sp);
    fprintf("Gaussian std ([0,1] scale): %.6f\n", Gaussian_std);
    fprintf("MSE0  = %.6f\n", Results0.MSE);
    fprintf("PSNR0 = %.4f dB\n", Results0.PSNR);
    fprintf("SSIM0 = %.4f\n\n", Results0.SSIM);

    %% ============================================================
    % STEP 1 — ADAPTIVE MEDIAN FILTER
    %% ============================================================
    fprintf("\n--- STEP 1: Adaptive Median Filter ---\n");
    I_step1 = adaptive_median_filtering(I_degraded, 5);

    Results1 = compare_original_restored(I_original, I_step1);
    fprintf("After Adaptive Median: PSNR=%.4f dB, SSIM=%.4f\n", Results1.PSNR, Results1.SSIM);

    show_compare_images(I_original, I_degraded, I_step1, ...
        "Step 1: Adaptive Median Filter");
    outdir = "";
    if ~exist(outdir,"dir"), mkdir(outdir); end
    imwrite(I_step1, fullfile(outdir, name + "_step1_adaptmed.png"));
    %% ============================================================
    % STEP 2 — SA-DCT BLIND DEBLUR + DENOISE (PSF UNKNOWN)
    %% ============================================================
    fprintf("\n--- STEP 2: SA-DCT Blind Deblur + Denoise (sadct_blind_deblur) ---\n");

    [I_step2] = perform_blsgsm_denoising(I_step1);  % returns same class as input (uint8)

    Results2 = compare_original_restored(I_original, I_step2);
    fprintf("After SA-DCT Blind: PSNR=%.4f dB, SSIM=%.4f | mode=%s\n", ...
        Results2.PSNR, Results2.SSIM);

    show_compare_images(I_original, I_step1, I_step2, ...
        "Step 2: SA-DCT Blind Deblur + Denoise (After Adaptive Median)");

    % Optional: visualize estimated PSF if blind path was used
    if isfield(info_sadct,'psf_est') && ~isempty(info_sadct.psf_est)
        try
            figure; imagesc(info_sadct.psf_est); axis image; colormap gray;
            title("Estimated PSF (SA-DCT blind)");
        catch
        end
    end

    %% ============================================================
    % STEP 3 — NLM AFTER SA-DCT (LIGHT CLEANUP)
    %% ============================================================
    fprintf("\n--- STEP 3: NLM (imnlmfilt) after SA-DCT ---\n");

    if exist("imnlmfilt","file") ~= 2
        warning("imnlmfilt not found. Skipping Step 3.");
        I_step3 = I_step2;
    else
        % Use SA-DCT estimated sigma (in [0,1]) to set a reasonable DegreeOfSmoothing
        if isfield(info_sadct,'sigma') && ~isempty(info_sadct.sigma)
            sig01 = double(info_sadct.sigma);
        else
            sig01 = std(im2double(I_step2(:)) - imgaussfilt(im2double(I_step2), 1.0), 0, "all"); % fallback
        end

        % Map sigma -> DegreeOfSmoothing (bounded, conservative)
        deg = round(5 + 400*sig01);          % sigma 0.005 -> ~7, 0.02 -> ~13, 0.05 -> ~25
        deg = min(30, max(5, deg));

        sw = 21;  % SearchWindowSize (odd)
        cw = 7;   % ComparisonWindowSize (odd), <= sw

        fprintf("NLM params: DegreeOfSmoothing=%d, SW=%d, CW=%d (sigma~%.6f)\n", deg, sw, cw, sig01);

        I_step3 = imnlmfilt(I_step2, ...
            "DegreeOfSmoothing", deg, ...
            "SearchWindowSize", sw, ...
            "ComparisonWindowSize", cw);
    end

    Results3 = compare_original_restored(I_original, I_step3);
    fprintf("After NLM post-SA-DCT: PSNR=%.4f dB, SSIM=%.4f\n", Results3.PSNR, Results3.SSIM);

    show_compare_images(I_original, I_step2, I_step3, ...
        "Step 3: NLM after SA-DCT");

    %% ============================================================
    % METRIC SUMMARY (THIS IMAGE)
    %% ============================================================
    Steps = {
        "Degraded", ...
        "Adaptive Median", ...
        "SA-DCT Blind Deblur", ...
        "NLM post-SA-DCT"
    };

    MSE_values  = [Results0.MSE,  Results1.MSE,  Results2.MSE,  Results3.MSE];
    PSNR_values = [Results0.PSNR, Results1.PSNR, Results2.PSNR, Results3.PSNR];
    SSIM_values = [Results0.SSIM, Results1.SSIM, Results2.SSIM, Results3.SSIM];

    SummaryTable = table(Steps', MSE_values', PSNR_values', SSIM_values', ...
        'VariableNames', {'Step','MSE','PSNR','SSIM'});

    fprintf("\n===== METRIC SUMMARY FOR IMAGE: %s =====\n", upper(name));
    disp(SummaryTable);

    %% ============================================================
    % STORE FINAL METRICS (AFTER STEP 3)
    %% ============================================================
    All_MSE  = [All_MSE;  Results3.MSE];
    All_PSNR = [All_PSNR; Results3.PSNR];
    All_SSIM = [All_SSIM; Results3.SSIM];

end

%% ============================================================
% FINAL SUMMARY ACROSS ALL IMAGES
%% ============================================================
FinalTable = table(image_names', All_MSE, All_PSNR, All_SSIM, ...
    'VariableNames', {'Image','Final_MSE','Final_PSNR','Final_SSIM'});

fprintf("\n=============================================\n");
fprintf("        FINAL SUMMARY (ALL IMAGES)\n");
fprintf("=============================================\n");
disp(FinalTable);

fprintf("MEAN Final PSNR = %.4f dB\n", mean(All_PSNR));
fprintf("MEAN Final SSIM = %.4f\n", mean(All_SSIM));
