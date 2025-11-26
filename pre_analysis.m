% ==========================
% Load Images
% ==========================
img1 = imread('project_images/degraded/degraded_boat.png');
img2 = imread('project_images/clean/original_boat.png');

% ==========================
% Compute FFTs
% ==========================
F1 = fft2(double(img1));
F1s = fftshift(F1);
mag1 = log(1 + abs(F1s));

F2 = fft2(double(img2));
F2s = fftshift(F2);
mag2 = log(1 + abs(F2s));

% ==========================
% Plot Results
% ==========================
figure;

% ------- Degraded -------
subplot(2,3,1);
imshow(img1, []);
title('Degraded Image');

subplot(2,3,2);
imshow(mag1, []);
title('Degraded FFT Spectrum');

subplot(2,3,3);
imhist(img1);
title('Degraded Histogram');

% ------- Original -------
subplot(2,3,4);
imshow(img2, []);
title('Original Image');

subplot(2,3,5);
imshow(mag2, []);
title('Original FFT Spectrum');

subplot(2,3,6);
imhist(img2);
title('Original Histogram');








% ==========================
% Load Images (Grayscale)
% ==========================
I_degraded = imread("project_images/degraded/degraded_baboon.png");
I_original = imread("project_images/clean/original_baboon.png");

% ==========================
% Apply Median Filter
% ==========================
I_med = medfilt2(I_degraded, [5 5]);   % 3x3 median filter

% ==========================
% Show Images (Degraded vs Median vs Clean)
% ==========================
figure;
subplot(1,3,1), imshow(I_degraded), title("Degraded Image")
subplot(1,3,2), imshow(I_med), title("Median Filtered Image")
subplot(1,3,3), imshow(I_original), title("Clean Original Image")

% ==========================
% Histogram Comparison (Separate Figure)
% ==========================
figure;
hold on;
imhist(I_degraded);   % degraded
imhist(I_med);        % median filtered
imhist(I_original);   % clean
hold off;

title("Histogram Comparison: Degraded vs Median Filtered vs Clean");
legend("Degraded", "Median Filtered", "Clean Original");





% ==========================
% Load Images (Grayscale)
% ==========================
I_degraded = imread("project_images/degraded/degraded_baboon.png");
I_original = imread("project_images/clean/original_baboon.png");

% ==========================
% Step 1: Gaussian Blur
% ==========================
I_gauss = imgaussfilt(I_degraded, 1);   % sigma = 1 (you can adjust)

% ==========================
% Step 2: Median Filter
% ==========================
I_med = medfilt2(I_gauss, [5 5]);      % 5x5 median filter

% ==========================
% Step 3: Sharpening Kernel
% ==========================
K = [0 -1  0;
    -1  5 -1;
     0 -1  0];

% Apply convolution
I_sharp = conv2(double(I_med), K, 'same');

% Clip to valid range [0,255]
I_sharp = uint8(max(0, min(255, I_sharp)));

% ==========================
% Show Results
% ==========================
figure;
subplot(1,4,1), imshow(I_degraded, []), title("Degraded Image")
subplot(1,4,2), imshow(I_gauss, []), title("Gaussian Blurred")
subplot(1,4,3), imshow(I_med, []), title("Median Filtered")
subplot(1,4,4), imshow(I_sharp, []), title("Sharpened (conv2)")







%% ============================================================
% 1. LOAD IMAGES (Grayscale)
% =============================================================
I_clean    = imread("project_images/clean/original_baboon.png");
I_degraded = imread("project_images/degraded/degraded_baboon.png");

I_clean    = uint8(I_clean);
I_degraded = uint8(I_degraded);


%% ============================================================
% 2. MEDIAN FILTER  (Salt & Pepper Removal)
% =============================================================
I_med = medfilt2(I_degraded, [5 5]);


%% ============================================================
% 3. WIENER FILTER (Gaussian Noise Removal)
% =============================================================
I_wien = wiener2(I_med, [5 5]);


%% ============================================================
% 4. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
% =============================================================
I_clahe = adapthisteq(I_wien, "ClipLimit", 0.015);


%% ============================================================
% 5. DEBLUR (Wiener Deconvolution with small Gaussian PSF)
% =============================================================
PSF   = fspecial('gaussian', 5, 1.0);   % mild blur removal
NSR   = 0.003;                          % noise-to-signal ratio estimate
I_deblur = deconvwnr(I_clahe, PSF, NSR);


%% ============================================================
% 6. SHARPEN (Unsharp Mask / Laplacian)
% =============================================================
K = [0 -1 0;
    -1 5 -1;
     0 -1 0];

I_sharp = conv2(double(I_deblur), K, 'same');
I_sharp = uint8(max(0, min(255, I_sharp)));


%% ============================================================
% 7. FINAL HISTOGRAM MATCHING (to clean baboon)
% =============================================================
I_final = imhistmatch(I_sharp, I_clean);


%% ============================================================
% 8. SHOW PIPELINE RESULTS
% =============================================================
figure;
subplot(1,7,1), imshow(I_degraded), title("Degraded");
subplot(1,7,2), imshow(I_med),      title("Median");
subplot(1,7,3), imshow(I_wien),     title("Wiener");
subplot(1,7,4), imshow(I_clahe),    title("CLAHE");
subplot(1,7,5), imshow(I_deblur),   title("Deblurred");
subplot(1,7,6), imshow(I_sharp),    title("Sharpened");
subplot(1,7,7), imshow(I_final),    title("Final Restored");


%% ============================================================
% 9. FINAL COMPARISONS WITH CLEAN
% =============================================================
figure;
subplot(2,3,1), imshow(I_final), title("Final Restored");
subplot(2,3,2), imshow(log(1+abs(fftshift(fft2(double(I_final))))), []), ...
                 title("Final FFT");
subplot(2,3,3), imhist(I_final), title("Final Histogram");

subplot(2,3,4), imshow(I_clean), title("Clean");
subplot(2,3,5), imshow(log(1+abs(fftshift(fft2(double(I_clean))))), []), ...
                 title("Clean FFT");
subplot(2,3,6), imhist(I_clean), title("Clean Histogram");




clc,clear all,close all

%% LOAD DEGRADED IMAGE
I_degraded = imread("project_images/degraded/degraded_boat.png");
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
I_clean = imread("project_images/clean/original_boat.png");
I_clean = uint8(I_clean);

I_gauss_sp = imnoise(I_clean, "gaussian", 0, 0.01);   % Gaussian noise
I_gauss_sp = imnoise(I_gauss_sp, "salt & pepper", P_sp);   % S&P noise


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





