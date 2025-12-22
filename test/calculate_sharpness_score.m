function score = calculate_sharpness_score(img)
% CALCULATE_SHARPNESS_SCORE Edge/Noise oranına dayalı keskinlik skoru
%   score = calculate_sharpness_score(img)
%
%   Bu fonksiyon görüntünün kalitesini ölçmek için kullanılır:
%   Score = Tenengrad / (σ + ε)
%
%   Tenengrad: Sobel gradient energy (kenar keskinliği)
%   σ: Laplacian-MAD noise estimation
%   ε: Küçük sabit (bölme hatasını önlemek için)
%
%   Yüksek skor = İyi (keskin kenarlar, düşük noise)
%   Düşük skor = Kötü (bulanık kenarlar veya yüksek noise)
%
%   Girdi:
%       img - Gri tonlamalı görüntü (uint8 veya double)
%
%   Çıktı:
%       score - Keskinlik skoru (yüksek = daha iyi)

    % Görüntüyü double'a çevir
    if ~isa(img, 'double')
        img = double(img);
    end
    
    epsilon = 1e-6;  % Bölme hatasını önlemek için
    
    % --- TENENGRAD (Sobel Gradient Energy) ---
    % Sobel filtreleri
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = Gx';
    
    % Gradient hesapla
    Ix = imfilter(img, Gx, 'replicate');
    Iy = imfilter(img, Gy, 'replicate');
    
    % Gradient magnitude karesi (Tenengrad)
    tenengrad = mean(Ix(:).^2 + Iy(:).^2);
    
    % --- NOISE ESTIMATION (Laplacian-MAD) ---
    sigma = estimate_gaussian_noise(img);
    
    % --- SCORE ---
    score = tenengrad / (sigma + epsilon);
end
