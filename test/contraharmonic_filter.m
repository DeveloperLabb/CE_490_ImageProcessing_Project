%% ============================================================
% HELPER: CONTRA-HARMONIC MEAN FILTER
%% ============================================================
function out = contraharmonic_filter(img, kernel_size, Q)
    img = double(img);
    pad = floor(kernel_size/2);

    img_pad = padarray(img, [pad pad], 'replicate');

    [M, N] = size(img);
    out = zeros(M, N);

    for i = 1:M
        for j = 1:N
            window = img_pad(i:i+2*pad, j:j+2*pad);

            num = sum(window.^(Q+1), 'all');
            den = sum(window.^Q, 'all');

            if den == 0
                out(i,j) = img(i,j); % fallback
            else
                out(i,j) = num / den;
            end
        end
    end

    out = uint8(max(0, min(255, out)));
end
