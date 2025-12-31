function I_out = edge_masked_laplacian_enhance(I_sharpened, I_base)
% EDGE_MASKED_LAPLACIAN_ENHANCE - Edge-masked Laplacian enhancement
%
% Girdi:
%   I_sharpened - Keskinleştirilmiş görüntü (uint8) - Laplacian hesaplanacak
%   I_base      - Baz görüntü (uint8) - Enhancement eklenecek
%
% Çıktı:
%   I_out       - Geliştirilmiş görüntü (uint8)

    fprintf("\n--- STEP 4: Edge-Masked Laplacian Enhancement ---\n");
    
    laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    I_sharpened_double = double(I_sharpened) / 255;
    laplace_output = imfilter(I_sharpened_double, laplacian_kernel, 'symmetric');

    enhanced_edges = double(laplace_output);

    [~, noise_sigma] = edge_noise_metrics(I_sharpened_double);
    noise_var = noise_sigma^2;

    lap_smooth_w = wiener2(enhanced_edges, [3 3], noise_var);

    meanKernel = fspecial('average', [3 3]);
    lap_smooth = imfilter(lap_smooth_w, meanKernel, 'symmetric');

    I_base_double = double(I_base) / 255;
    I_out_double = I_base_double + lap_smooth;

    I_out_double = max(0, min(1, I_out_double));
    I_out = im2uint8(I_out_double);
end
