function y_hat = adaptive_sharpen(z, amount, radius)
% ADAPTIVE_SHARPEN - Edge-aware adaptive sharpening
%
% Sharpens blurred images using unsharp masking with edge-aware
% blending to prevent noise amplification in flat regions.
%
% Inputs:
%   z       - Blurred image (uint8 or double)
%   amount  - Sharpening strength (0.5 to 3.0, default 1.0)
%   radius  - Blur radius for mask (0.5 to 3.0, default 1.0)
%
% Output:
%   y_hat   - Sharpened image

    % Convert to double [0, 1]
    was_uint8 = isa(z, 'uint8');
    z = im2double(z);
    
    %% ========================================
    % STEP 1: Create Blurred Version (Low-pass)
    %% ========================================
    
    % Gaussian blur for unsharp mask
    sigma = radius;
    z_blur = imgaussfilt(z, sigma);
    
    %% ========================================
    % STEP 2: Compute High-Frequency Detail
    %% ========================================
    
    % High-frequency component = original - blurred
    high_freq = z - z_blur;
    
    %% ========================================
    % STEP 3: Edge-Aware Sharpening Mask
    %% ========================================
    
    % Detect edges using gradient magnitude
    [Gx, Gy] = imgradientxy(z, 'sobel');
    grad_mag = sqrt(Gx.^2 + Gy.^2);
    
    % Normalize gradient to [0, 1]
    grad_mag = grad_mag / (max(grad_mag(:)) + eps);
    
    % Edge-aware mask: sharpen more near edges, less in flat regions
    % This prevents noise amplification in flat areas
    edge_mask = grad_mag;
    
    % Smooth the mask to avoid harsh transitions
    edge_mask = imgaussfilt(edge_mask, 2);
    
    % Scale mask to [0.3, 1.0] - always apply some sharpening, more at edges
    edge_mask = 0.3 + 0.7 * edge_mask;
    
    %% ========================================
    % STEP 4: Apply Adaptive Sharpening
    %% ========================================
    
    % Sharpened = Original + amount * high_freq * mask
    y_sharpened = z + amount * high_freq .* edge_mask;
    
    %% ========================================
    % STEP 5: Noise-Aware Blending
    %% ========================================
    
    % Estimate local noise using variance in small windows
    % High variance in flat areas indicates noise
    local_std = stdfilt(z, ones(5));
    
    % Normalize
    local_std = local_std / (max(local_std(:)) + eps);
    
    % Create noise mask: blend towards original in noisy flat regions
    % Low gradient + high std = noisy flat region
    noise_mask = (1 - grad_mag) .* local_std;
    noise_mask = imgaussfilt(noise_mask, 3);
    
    % Blend: use sharpened where edges, original where noisy-flat
    blend_weight = 1 - 0.5 * noise_mask;  % 0.5 to 1.0
    y_hat = y_sharpened .* blend_weight + z .* (1 - blend_weight);
    
    %% ========================================
    % STEP 6: Clip and Convert
    %% ========================================
    
    % Ensure valid range
    y_hat = max(0, min(1, y_hat));
    
    % Convert back to uint8 if needed
    if was_uint8
        y_hat = im2uint8(y_hat);
    end
end
