function Results = compare_original_restored(I_original, I_restored)
%COMPARE_ORIGINAL_RESTORED
%   Computes MSE, PSNR, and SSIM between:
%       - Original
%       - Restored
%
%   Returns a table with metrics.

    % Convert images to double
    Io = double(I_original);
    Ir = double(I_restored);

    % --- MSE ---
    mse = @(A,B) mean((A(:) - B(:)).^2);

    % --- PSNR ---
    psnr_calc = @(A,B) 10 * log10(255^2 / mse(A,B));

    % --- SSIM ---
    ssim_val = ssim(uint8(I_restored), uint8(I_original));

    % Compute metrics
    MSE_val  = mse(Io, Ir);
    PSNR_val = psnr_calc(Io, Ir);

    % Return results as table
    Results = table(MSE_val, PSNR_val, ssim_val, ...
        'VariableNames', {'MSE','PSNR','SSIM'}, ...
        'RowNames', {'Restored'});
end
