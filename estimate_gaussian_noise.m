function sigma = estimate_gaussian_noise(img)
% Laplacian-MAD noise estimation: sigma ≈ median(|∇²I|) / (0.6745 * sqrt(20))

    if ~isa(img, 'double')
        img = double(img);
    end
    
    laplacian_kernel = [0  1 0; 
                        1 -4 1; 
                        0  1 0];
    
    laplacian_img = imfilter(img, laplacian_kernel, 'replicate');
    abs_laplacian = abs(laplacian_img(:));
    
    mad_value = median(abs_laplacian);
    normalization_factor = 0.6745 * sqrt(20);
    
    sigma = mad_value / normalization_factor;
end
