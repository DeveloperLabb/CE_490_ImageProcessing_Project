clc; clear all; close all;

%% LOAD IMAGES
I_original = uint8(imread("project_images/clean/original_baboon.png"));
I_degraded = uint8(imread("project_images/degraded/degraded_baboon.png"));


%% ============================================================
% STEP 0 — ORIGINAL vs DEGRADED NOISE MEASUREMENT
%% ============================================================
disp("=== STEP 0: Noise Analysis ===");

O = double(I_original);
D = double(I_degraded);

% ---- Salt & Pepper Noise Estimation ----
num_salt   = sum(D(:) == 255);
num_pepper = sum(D(:) == 0);
P_sp = (num_salt + num_pepper) / numel(D);
fprintf("Salt-Pepper Noise Ratio: %.5f\n", P_sp);

% ---- Gaussian Noise Estimation ----
Gaussian_std = std((D(:) - O(:)));
fprintf("Estimated Gaussian Noise Std Dev: %.3f\n", Gaussian_std);

% ---- MSE / PSNR / SSIM ----
MSE0  = mean((O(:) - D(:)).^2);
PSNR0 = 10*log10(255^2 / MSE0);
SSIM0 = ssim(I_degraded, I_original);

fprintf("MSE(Original, Degraded) = %.2f\n", MSE0);
fprintf("PSNR(Original, Degraded) = %.2f dB\n", PSNR0);
fprintf("SSIM(Original, Degraded) = %.4f\n\n", SSIM0);

% ---- Noise Map ----
figure;
subplot(1,3,1); imshow(I_original); title("Original");
subplot(1,3,2); imshow(I_degraded); title("Degraded");
subplot(1,3,3); imshow(D-O, []); title("Noise Map (Degraded - Original)");

% ---- FFT Comparison ----
F_orig = log(1+abs(fftshift(fft2(O))));
F_degr = log(1+abs(fftshift(fft2(D))));

figure;
subplot(1,2,1); imshow(F_orig, []); title("FFT Original");
subplot(1,2,2); imshow(F_degr, []); title("FFT Degraded");

% ---- Histogram Comparison ----
figure; hold on;
imhist(I_original);
imhist(I_degraded);
legend("Original","Degraded");
title("Histogram: Original vs Degraded");
hold off;



%% ============================================================
% STEP 1 — APPLY NORMAL MEDIAN FILTER  (Salt-Pepper Removal)
%% ============================================================
I_step1 = adaptive_median_filtering(I_degraded, 5);

disp("=== STEP 1: Normal Median Filter ===");
Results1 = compare_original_restored(I_original, I_step1)

show_compare_images(I_original, I_degraded, I_step1, ...
    "Step 1: Normal Median Filter");


%% ============================================================
% STEP 2 — APPLY GAUSSIAN SMOOTHING (Mild Gaussian Noise Removal)
%% ============================================================
I_step2 = gaussian_smoothing(I_step1, 3, 0.66);

disp("=== STEP 2: Gaussian Smoothing ===");
Results2 = compare_original_restored(I_original, I_step2)

show_compare_images(I_original, I_step1, I_step2, ...
    "Step 2: Gaussian Smoothing");


%% ============================================================
% STEP 2.5 — SOBEL FILTER (compute after smoothing)
%% ============================================================
[Gx, Gy, Gmag] = sobel_filter(I_step2);

disp("=== STEP 2.5: Sobel Edges Computed After Step 2 ===");

figure;
subplot(1,3,1), imshow(Gx), title("Sobel Gx");
subplot(1,3,2), imshow(Gy), title("Sobel Gy");
subplot(1,3,3), imshow(Gmag), title("Sobel Gradient Magnitude");


%% ============================================================
% STEP 3 — HYBRID GAUSSIAN + BUTTERWORTH HIGH-BOOST SHARPEN
%% ============================================================
k1 = 1.2;      
sigma_f = 25;  
k2 = 1.0;      
n = 2;         
D0 = 40;       

I_step3 = fft_hybrid_sharpen(I_step2, k1, sigma_f, k2, n, D0);

disp("=== STEP 3: Hybrid Gaussian + Butterworth High-Boost Sharpen ===");
Results3 = compare_original_restored(I_original, I_step3)

show_compare_images(I_original, I_step2, I_step3, ...
    "Step 3: Hybrid Gaussian + Butterworth Sharpen");


%% ============================================================
% STEP 4 — BILATERAL FILTER (Reduce sharpen-induced noise)
%% ============================================================
I_step4 = imbilatfilt(I_step3, 200, 15);

disp("=== STEP 4: Bilateral Filter (Sharpen Noise Suppression) ===");
Results4 = compare_original_restored(I_original, I_step4)

show_compare_images(I_original, I_step3, I_step4, ...
    "Step 4: Bilateral Post-Smoothing");


%% ============================================================
% STEP 5 — FINAL: Combine Bilateral Output + Sobel Edges
%% ============================================================
alpha = 0.15;   % Edge enhancement weight

I_step5 = double(I_step4) + alpha * double(Gmag);
I_step5 = uint8(max(0, min(255, I_step5)));

disp("=== STEP 5: Final Sobel-Enhanced Output ===");
Results5 = compare_original_restored(I_original, I_step5)

show_compare_images(I_original, I_step4, I_step5, ...
    "Step 5: Final Sobel + Bilateral Combined Output");
