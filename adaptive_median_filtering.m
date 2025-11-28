function output = adaptive_median_filtering(img, fSzMax)

    if mod(fSzMax, 2) == 0 || fSzMax < 3
        error('fSzMax must be an odd integer >= 3');
    end

    [rows, cols] = size(img);

    output = zeros(rows, cols, 'like', img);

    pSzMax = (fSzMax - 1) / 2;

    padded_img = padarray(img, [pSzMax pSzMax], 'symmetric');

    for i = 1:rows
        for j = 1:cols

            x = i + pSzMax;
            y = j + pSzMax;

            S = 3;                  
            pixel_processed = false;

            while S <= fSzMax && ~pixel_processed

                k = (S - 1) / 2;
                region = padded_img(x-k:x+k, y-k:y+k);

                z_min = min(region(:));
                z_max = max(region(:));
                z_med = median(region(:));
                z_xy  = padded_img(x, y);

                % -------- LEVEL A --------
                A1 = z_med - z_min;
                A2 = z_med - z_max;

                if (A1 > 0) && (A2 < 0)
                    % Median is NOT an impulse. Go to Level B.

                    % -------- LEVEL B --------
                    B1 = z_xy - z_min;
                    B2 = z_xy - z_max;

                    if (B1 > 0) && (B2 < 0)
                        % Original pixel is clean.
                        output(i, j) = z_xy;
                    else
                        % Original pixel is noise. Use median.
                        output(i, j) = z_med;
                    end

                    pixel_processed = true;

                else
                    % Median IS an impulse (or window is uniform).
                    % Increase window size and try again.
                    S = S + 2;

                    % If the window would exceed the maximum allowed size,
                    % accept the last median as the best estimate.
                    if S > fSzMax
                        output(i, j) = z_med;  % z_med from previous window
                        pixel_processed = true;
                    end
                end
            end
        end
    end

    % Match input class explicitly (optional if you used 'like' above)
    output = cast(output, 'like', img);
end

