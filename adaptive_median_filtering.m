function output = adaptive_median_filtering(input, fSzMax)

[rows, cols] = size(input);

output = zeros(rows,cols);

pSzMax = (fSzMax+1) / 2;

padded_img = padarray(input, [pSzMax pSzMax], 'symmetric');

for i = 1:rows
    for j = 1:cols
        x = i+pSzMax;
        y = j+pSzMax;
        
        S = 3;
        pixel_processed = false;
        
        while S <= pSzMax && pixel_processed == false
            
            k = (S+1) / 2;
            region = padded_img(x-k:x+k, y-k:y+k);

            z_min = min(region(:));
            z_max = max(region(:));
            z_med = median(region(:));

            z_xy  = padded_img(x, y);

            A1 = z_med - z_min;
            A2 = z_med - z_max;

            if (A1 > 0) && (A2 < 0)
                    % Median is NOT an impulse. Go to Level B.
                    
                    % --- LEVEL B ---
                    % Check if the original center pixel is an impulse
                    B1 = z_xy - z_min;
                    B2 = z_xy - z_max;
                    
                    if (B1 > 0) && (B2 < 0)
                        % The original pixel is clean (not an impulse).
                        output(i, j) = z_xy;
                    else
                        % The original pixel is noise. Replace with median.
                        output(i, j) = z_med;
                    end
                    
                    pixel_processed = true;
                    
                else
                    % Median IS an impulse (or window is uniform).
                    % Increase window size and try again.
                    S = S + 2;
                    
                    % If the window gets too large, we accept the median
                    % as the best guess to prevent infinite looping or crashing.
                    if S > pSzMax
                        output(i, j) = z_med;
                        pixel_processed = true;
                    end
                end
        end
    end
end

output = uint8(output);
end
