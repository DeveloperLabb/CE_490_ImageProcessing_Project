function I_sharp = usm_edge_compress(I_in, amount, radius, threshold)

I = double(I_in);

% Gaussian blur (base)
base = imgaussfilt(I, radius);

% detail component
detail = I - base;

% threshold → weak noise removal
mask = abs(detail) > threshold;

% normalize to [-1,1]
detail_norm = detail ./ (max(abs(detail(:))) + eps);

% soft compress strong edges (anti-halo)
detail_soft = detail_norm .* (1 - 0.35 * abs(detail_norm));

% sharpen only masked areas
I_sharp = I + amount * 255 * detail_soft .* mask;

% clip
I_sharp = uint8(max(0, min(255, I_sharp)));

end
