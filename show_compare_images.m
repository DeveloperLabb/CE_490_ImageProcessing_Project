function show_compare_images(I_original, I_degraded, I_step, stepName)
%SHOW_COMPARE_IMAGES
%   Displays original, degraded, and restored(step) images side-by-side.
%
%   INPUTS:
%       I_original  → Clean/original reference image
%       I_degraded  → Noisy/degraded image
%       I_step      → Output image after some restoration step
%       stepName    → Title string (ex: "Median Filter Step")

    figure;
    subplot(1,3,1), imshow(I_original), title("Original");
    subplot(1,3,2), imshow(I_degraded), title("Degraded");
    subplot(1,3,3), imshow(I_step),     title(stepName);

end
