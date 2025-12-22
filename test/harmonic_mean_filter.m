function out = harmonic_mean_filter(img, kernel_size)

img = double(img);
pad = floor(kernel_size/2);
img_pad = padarray(img, [pad pad], 'replicate');

[M, N] = size(img);
out = zeros(M, N);

for i = 1:M
    for j = 1:N
        window = img_pad(i:i+2*pad, j:j+2*pad);

        % Prevent division by zero
        window(window == 0) = 1e-6;

        H = kernel_size^2 / sum(1 ./ window, 'all');
        out(i,j) = H;
    end
end

out = uint8(out);
end
