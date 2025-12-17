function [edgeStrength, noiseFlat] = edge_noise_metrics(Iu8)
% Returns:
%   edgeStrength: mean gradient magnitude (Tenengrad-like)
%   noiseFlat:    Laplacian roughness measured only on low-gradient pixels

    I = im2double(Iu8);

    % --- Sobel gradients ---
    sx = fspecial('sobel');
    sy = sx';
    Gx = imfilter(I, sx, 'replicate', 'conv');
    Gy = imfilter(I, sy, 'replicate', 'conv');
    G  = hypot(Gx, Gy);

    edgeStrength = mean(G(:));

    % --- Flat-region mask: bottom 30% of gradient magnitudes ---
    thr = prctile(G(:), 30);
    flatMask = (G <= thr);

    % --- Laplacian roughness (noise/artifact proxy) in flat areas ---
    lapKernel = [0 -1 0; -1 4 -1; 0 -1 0];
    L = imfilter(I, lapKernel, 'replicate', 'conv');

    vals = L(flatMask);
    noiseFlat = std(vals(:));   % robust alternative: median(abs(vals))/0.6745
end
