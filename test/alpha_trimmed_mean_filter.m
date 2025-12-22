function J = alpha_trimmed_mean_filter(I, win, alpha)
%ALPHA_TRIMMED_MEAN_FILTER Alpha-trimmed mean filtering for grayscale images.
%   I     : uint8/uint16/single/double grayscale image
%   win   : odd window size (e.g., 3,5,7,...)
%   alpha : fraction of samples trimmed in TOTAL (0.. <1)
%           Half is removed from low end and half from high end.

    arguments
        I
        win (1,1) {mustBeInteger, mustBePositive}
        alpha (1,1) double {mustBeGreaterThanOrEqual(alpha,0), mustBeLessThan(alpha,1)}
    end

    if mod(win,2) == 0
        error("win must be odd (e.g., 3,5,7,...)");
    end

    I_in_class = class(I);
    Id = double(I);

    pad = floor(win/2);
    Ip  = padarray(Id, [pad pad], "symmetric");

    % Sliding blocks as columns: (win*win) x (H*W)
    cols = im2col(Ip, [win win], "sliding");
    cols = sort(cols, 1, "ascend");

    N = win * win;

    % total number of trimmed samples
    t_total = floor(alpha * N);
    t_total = min(t_total, N-1);        % keep at least 1 sample
    t_total = t_total - mod(t_total,2); % make it even
    t = t_total / 2;                    % per-tail trim count

    cols_trim = cols((1+t):(N-t), :);
    m = mean(cols_trim, 1);

    Jd = reshape(m, size(Id));

    % cast back nicely
    if isinteger(I)
        Jd = max(min(Jd, double(intmax(I_in_class))), double(intmin(I_in_class)));
        J  = cast(round(Jd), I_in_class);
    else
        J = cast(Jd, I_in_class);
    end
end
