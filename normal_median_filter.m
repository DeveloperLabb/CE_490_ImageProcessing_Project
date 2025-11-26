%% 1. NORMAL MEDIAN FILTER (apply to entire image)
function I_out = normal_median_filter(I_in, windowSize)

    I = double(I_in);
    pad = floor(windowSize/2);

    % Pad image (symmetric)
    I_pad = padarray(I, [pad pad], 'symmetric');

    I_out = zeros(size(I));

    for r = 1:size(I,1)
        for c = 1:size(I,2)

            % Extract window
            local = I_pad(r : r+2*pad, c : c+2*pad);

            % Apply median to all pixels
            I_out(r,c) = median(local(:));
        end
    end

    I_out = uint8(I_out);
end
