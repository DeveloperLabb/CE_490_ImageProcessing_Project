function ratio = measure_salt_pepper_ratio(img)
% MEASURE_SALT_PEPPER_RATIO Salt and pepper noise oranını ölçer
%   ratio = measure_salt_pepper_ratio(img)
%
%   Girdi:
%       img - Gri tonlamalı görüntü (uint8 veya double [0-255])
%
%   Çıktı:
%       ratio - Salt (255) ve pepper (0) piksellerinin toplam oranı
%               Değer 0 ile 1 arasında döner
%
%   Örnek:
%       noisy_img = imnoise(img, 'salt & pepper', 0.05);
%       ratio = measure_salt_pepper_ratio(noisy_img);
%       fprintf('Salt & Pepper Noise Oranı: %.4f (%%%.2f)\n', ratio, ratio*100);

    % Görüntüyü double'a çevir (eğer değilse)
    if ~isa(img, 'double')
        img = double(img);
    end
    
    % Toplam piksel sayısı
    total_pixels = numel(img);
    
    % Salt (255) ve Pepper (0) piksel sayıları
    salt_count = sum(img(:) == 255);
    pepper_count = sum(img(:) == 0);
    
    % Toplam salt & pepper piksel sayısı
    noise_pixels = salt_count + pepper_count;
    
    % Oran hesapla
    ratio = noise_pixels / total_pixels;
end
