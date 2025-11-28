function I_sharp = guided_sharpen(I_in, window_size, eps, alpha)

    % Convert to double
    I = double(I_in);

    % imguidedfilter requires window_size to be odd integer
    if mod(window_size,2)==0
        error("window_size MUST be an odd integer (3,5,7...).");
    end

    % Guided filter
    I_gf = imguidedfilter(I_in, ...
        'NeighborhoodSize', [window_size window_size], ...
        'DegreeOfSmoothing', eps);

    % Raw edges
    edge_raw = I - double(I_gf);

    % Normalize edges
    edge_norm = edge_raw ./ (max(abs(edge_raw(:))) + eps);

    % Compress
    edge_soft = edge_norm .* (1 - 0.4*abs(edge_norm));

    % Add edge
    I_sharp = I + alpha * 255 * edge_soft;

    I_sharp = uint8(max(0, min(255, I_sharp)));

end
