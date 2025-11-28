
clc,clear all,close all

%% LOAD DEGRADED IMAGE
I_degraded = imread("project_images/degraded/degraded_barbara.png");
I_degraded = uint8(I_degraded);

%% FIND SALT (255) AND PEPPER (0) PIXELS
pepper_mask = (I_degraded == 0);
salt_mask   = (I_degraded == 255);

num_pepper = sum(pepper_mask(:));
num_salt   = sum(salt_mask(:));

total_pixels = numel(I_degraded);

P_sp = (num_pepper + num_salt) / total_pixels;   % salt & pepper ratio

fprintf("Detected Salt & Pepper ratio in degraded image = %.5f\n", P_sp);

%% NOW USE THIS VALUE IN NOISE GENERATION
I_clean = imread("project_images/clean/original_barbara.png");
I_clean = uint8(I_clean);
% I_clean = conv2(I_clean,ones(3,3)/9,"same");

I_clean = imgaussfilt(I_clean,1.75);

I_gauss_sp = imnoise(I_clean, "gaussian", 0, 0.01);   % Gaussian noise
I_gauss_sp = imnoise(I_gauss_sp, "salt & pepper", 0.0175);   % S&P noise

figure;
imshow(uint8(I_gauss_sp));
figure;
imshow(uint8(I_degraded));

%% ============================================================
% APPLY MEDIAN FILTER TO BOTH IMAGES
% ============================================================
I_degraded_med  = medfilt2(I_degraded, [5 5]);   % Real degraded → median
I_synthetic_med = medfilt2(I_gauss_sp, [5 5]);   % Synthetic noisy → median


%% HISTOGRAM COMPARISON AFTER MEDIAN
figure; hold on;
imhist(I_synthetic_med);
imhist(I_degraded_med);
legend("Synthetic + Median", "Real Degraded + Median");
title("Histogram Comparison After Median Filtering");
hold off;


%% ============================================================
% PERCENTAGE ERROR BASED ON MAX INTENSITY (255)
% ============================================================
I_syn  = double(I_synthetic_med);
I_real = double(I_degraded_med);

AbsDiff = abs(I_real - I_syn);

PercentErrorMap = (AbsDiff / 255) * 100;   % Pixel-wise % error

MeanPercentError = mean(PercentErrorMap(:));

fprintf("\nMean Percentage Error After Median Filter = %.2f %%\n", ...
        MeanPercentError);


%% OPTIONAL VISUAL DIFFERENCE MAP (After Median)
figure;
imshow(abs(I_real - I_syn), []);
title("Absolute Error Map After Median (|Real - Synthetic|)");
