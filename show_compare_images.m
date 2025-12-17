function show_compare_images(I_original, I_degraded, I_step, stepName)
%SHOW_COMPARE_IMAGES
% 3 images on the top row + 2 FFT magnitude spectra on the bottom row.

    % --- Ensure grayscale doubles for FFT ---
    O = double(I_original);
    S = double(I_step);

    % --- FFT magnitude (centered + log) ---
    F_O = fftshift(fft2(O));
    F_S = fftshift(fft2(S));

    Mag_O = mat2gray(log(1 + abs(F_O)));
    Mag_S = mat2gray(log(1 + abs(F_S)));

    figure;

    % ===== Top row (3 images) =====
    subplot(2,3,1), imshow(I_original), title("Original");
    subplot(2,3,2), imshow(I_degraded), title("Degraded");
    subplot(2,3,3), imshow(I_step),     title(stepName);

    % ===== Bottom row (2 FFTs, larger) =====
    subplot(2,3,4), imshow(Mag_O), title("FFT |Original| (log)");
    subplot(2,3,5), imshow(Mag_S), title("FFT |Step| (log)");

    % Leave subplot(2,3,6) empty on purpose for readability
    subplot(2,3,6), axis off;
end
