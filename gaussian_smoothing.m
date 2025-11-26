function I_out = gaussian_smoothing(I_in, kernelSize, sigma)
%GAUSSIAN_SMOOTHING
%   Applies Gaussian smoothing using a manually constructed kernel.
%
%   I_out = gaussian_smoothing(I_in, kernelSize, sigma)
%
%   Example:
%       I_blur = gaussian_smoothing(I, 5, 1.0);

    % Convert to double for filtering
    I = double(I_in);

    % Ensure odd kernel size
    if mod(kernelSize,2) == 0
        error("kernelSize must be odd (3,5,7,...)");
    end

    % Half-size for indexing
    k = floor(kernelSize/2);

    % Generate Gaussian kernel manually
    [x, y] = meshgrid(-k:k, -k:k);
    G = exp(-(x.^2 + y.^2) / (2*sigma^2));
    G = G / sum(G(:));    % normalize kernel

    % Convolve
    I_blur = conv2(I, G, 'same');

    % Convert back to uint8
    I_out = uint8(max(0, min(255, I_blur)));

end
