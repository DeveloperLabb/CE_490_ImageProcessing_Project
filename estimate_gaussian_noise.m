function sigma = estimate_gaussian_noise(img)
% ESTIMATE_GAUSSIAN_NOISE Laplacian-MAD yöntemiyle Gaussian noise std tahmini
%   sigma = estimate_gaussian_noise(img)
%
%   Bu yöntem akademik literatürde en çok kullanılan noise estimation
%   tekniğidir. Laplacian filtresi ile yüksek frekans bileşenlerini alır,
%   Median Absolute Deviation (MAD) ile robust bir std tahmini yapar.
%
%   Formül: sigma ≈ median(|∇²I|) / 0.6745
%
%   Girdi:
%       img - Gri tonlamalı görüntü (uint8 veya double)
%
%   Çıktı:
%       sigma - Tahmini Gaussian noise standart sapması
%
%   Referans:
%       Donoho, D.L. (1995). "De-noising by soft-thresholding"
%       IEEE Trans. Information Theory, 41(3), 613-627.

    % Görüntüyü double'a çevir
    if ~isa(img, 'double')
        img = double(img);
    end
    
    % 3x3 Laplacian kernel (standart 8-connected)
    laplacian_kernel = [0  1 0; 
                        1 -4 1; 
                        0  1 0];
    
    % Laplacian uygula (kenarları replicate ile doldur)
    laplacian_img = imfilter(img, laplacian_kernel, 'replicate');
    
    % Mutlak değer al
    abs_laplacian = abs(laplacian_img(:));
    
    % MAD (Median Absolute Deviation) ile robust std tahmini
    % Gaussian dağılım için MAD -> std dönüşümü: sigma = MAD / 0.6745
    % Laplacian için ek normalizasyon faktörü gerekli (~0.6745 * sqrt(20))
    % Basitleştirilmiş formül: sigma ≈ median(|∇²I|) / 0.6745
    
    mad_value = median(abs_laplacian);
    
    % Normalizasyon faktörü (Laplacian kernel'ın etkisini telafi etmek için)
    % sqrt(20) Laplacian kernel'ın L2 norm'undan gelir
    normalization_factor = 0.6745 * sqrt(20);
    
    sigma = mad_value / normalization_factor;
end
