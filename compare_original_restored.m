function Results = compare_original_restored(I_original, I_restored)
%COMPARE_ORIGINAL_RESTORED (0-1 scale)
% Computes MSE/PSNR/SSIM on normalized [0,1] images.
% MSE is normalized (0..1).

    if ~isequal(size(I_original), size(I_restored))
        error("Input images must have the same size.");
    end

    % Convert both to double in [0,1]
    Io = im2double(I_original);
    Ir = im2double(I_restored);

    % Built-in metrics in 0-1 scale
    MSE_val  = immse(Ir, Io);                         % 0-1 scale MSE
    PSNR_val = psnr(Ir, Io, 1);                       % peak=1
    SSIM_val = ssim(Ir, Io, "DynamicRange", 1);       % range=1

    Results = table(MSE_val, PSNR_val, SSIM_val, ...
        'VariableNames', {'MSE','PSNR','SSIM'}, ...
        'RowNames', {'Restored'});
end
