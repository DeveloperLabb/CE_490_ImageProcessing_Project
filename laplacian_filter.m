function I_out = laplacian_filter(I_in, type)
%LAPLACIAN_FILTER Applies Laplacian or LoG filter
%
% type options:
%   "lap3"   → 3x3 standard Laplacian
%   "lap8"   → 3x3 8-neighbor Laplacian
%   "LoG"    → 5x5 Laplacian of Gaussian

    I = double(I_in);

    switch type
        case "lap3"
            K = [0 -1 0;
                -1 4 -1;
                 0 -1 0];

        case "lap8"
            K = [-1 -1 -1;
                 -1  8 -1;
                 -1 -1 -1];

        case "LoG"
            K = [0  0 -1  0  0;
                 0 -1 -2 -1  0;
                -1 -2 16 -2 -1;
                 0 -1 -2 -1  0;
                 0  0 -1  0  0];

        otherwise
            error("Unknown Laplacian type.");
    end

    % Convolution
    I_lap = conv2(I, K, 'same');

    % Normalize and clip
    I_out = uint8(max(0, min(255, I_lap)));
end
