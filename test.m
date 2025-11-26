clc; clear all; close all;

img_names = {'cameraman', 'peppers', 'baboon', 'boat', 'barbara'};

% Metrikleri Saklamak İçin Tablo Başlıkları
fprintf('%-12s | %-12s | %-12s | %-12s | %-12s\n', ...
    'Image', 'Init PSNR', 'Final PSNR', 'Init SSIM', 'Final SSIM');
fprintf('-------------------------------------------------------------------\n');

avg_psnr_gain = 0;

for k = 1:length(img_names)
    name = img_names{k};
    
    % --- 1. Yükleme ---
    try
        I_orig = imread(fullfile('project_images/clean', ['original_' name '.png']));
        I_deg  = imread(fullfile('project_images/degraded', ['degraded_' name '.png']));
    catch
        continue; 
    end
    
    if size(I_orig,3)==3, I_orig = rgb2gray(I_orig); end
    if size(I_deg,3)==3,  I_deg  = rgb2gray(I_deg); end

    % --- Initial Metrics (Başlangıç Durumu) ---
    mse_init = immse(I_deg, I_orig);
    psnr_init = 10*log10(255^2 / mse_init);
    ssim_init = ssim(I_deg, I_orig);

    % ============================================================
    %                  OPTIMIZED RESTORATION PIPELINE
    % ============================================================

    % ADIM 1: Salt & Pepper Temizliği (Adaptive Median)
    % S&P oranı düşük (%3-7) olduğu için 3x3 pencere yeterli ve detayı korur.
    % Fonksiyonunuz yoksa: I_step1 = medfilt2(I_deg, [3 3]);
    try
        I_step1 = adaptive_median_filtering(I_deg, 5); 
    catch
        I_step1 = medfilt2(I_deg, [3 3]);
    end

    % ADIM 2: Gaussian Temizliği (Wiener Filter)
    % Gaussian Smoothing yerine Wiener kullanıyoruz.
    % Wiener, yerel varyansa (local variance) bakar. 
    % [3 3] penceresi Baboon gibi detaylı resimler için kritiktir.
    I_step2 = wiener2(I_step1, [3 3]);

    % ADIM 3: Guided Filter (Bilateral yerine)
    % Bilateral bazen dokuyu "plastikleştirir". Guided Filter ise
    % orijinal resmin yapısını (guidance) kullanarak gürültü siler.
    % DegreeOfSmoothing parametresi gürültü seviyesine göre ayarlanır.
    I_step3 = imguidedfilter(I_step2, 'DegreeOfSmoothing', 0.002 * 255^2);

    % ADIM 4: Unsharp Masking (Restoration)
    % Blur etkisini kırmak için hafif keskinleştirme.
    % Radius 1'den büyük olursa Baboon bozulur.
    I_final = imsharpen(I_step3, 'Radius', 1, 'Amount', 1.0, 'Threshold', 0.05);

    % ============================================================
    
    % --- Final Metrics ---
    mse_final = immse(I_final, I_orig);
    psnr_final = 10*log10(255^2 / mse_final);
    ssim_final = ssim(I_final, I_orig);
    
    avg_psnr_gain = avg_psnr_gain + (psnr_final - psnr_init);

    fprintf('%-12s | %-10.2f dB | %-10.2f dB | %-10.4f   | %-10.4f\n', ...
        name, psnr_init, psnr_final, ssim_init, ssim_final);
end

fprintf('-------------------------------------------------------------------\n');
fprintf('AVERAGE PSNR GAIN: +%.2f dB per image\n', avg_psnr_gain / length(img_names));