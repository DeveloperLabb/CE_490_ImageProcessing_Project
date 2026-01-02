function [I_out, final_noise, total_iterations] = iterative_nlm_filter(I_in, gaussian_threshold, max_iterations, degree_list, sw_list, cw_list, step_name)
% Iterative Non-Local Means filtering for Gaussian noise reduction.

    I_out = I_in;
    gaussian_current = estimate_gaussian_noise(I_out);
    nlm_iteration = 0;

    fprintf("\n--- %s: Iterative NLM (Gaussian Noise Reduction) ---\n", step_name);
    fprintf("Target Gaussian Threshold: %.2f\n", gaussian_threshold);
    fprintf("Initial Gaussian Noise (Est.): %.3f\n\n", gaussian_current);

    while gaussian_current > gaussian_threshold && nlm_iteration < max_iterations
        nlm_iteration = nlm_iteration + 1;

        fprintf("  [NLM Iteration %d] - Current Noise: %.3f\n", nlm_iteration, gaussian_current);
        fprintf("  Grid Search başlatılıyor...\n");

        best_noise     = inf;
        best_desc      = "";
        best_candidate = I_out;

        for deg = degree_list
            for sw = sw_list
                for cw = cw_list
                    if cw > sw, continue; end

                    candidate = imnlmfilt(I_out, ...
                        "DegreeOfSmoothing", deg, ...
                        "SearchWindowSize", sw, ...
                        "ComparisonWindowSize", cw);

                    noise_est = estimate_gaussian_noise(candidate);

                    fprintf("    deg=%d SW=%d CW=%d -> Noise=%.3f\n", deg, sw, cw, noise_est);

                    if noise_est < best_noise
                        best_noise     = noise_est;
                        best_desc      = sprintf("deg=%d,SW=%d,CW=%d", deg, sw, cw);
                        best_candidate = candidate;
                    end
                end
            end
        end

        if best_noise >= gaussian_current
            fprintf("\n    >>> No improvement found (Best: %.3f >= Current: %.3f)\n", ...
                best_noise, gaussian_current);
            fprintf("    >>> Early stopping - NLM iteration terminated.\n\n");
            break;
        end

        fprintf("\n    >>> Best: %s | Noise: %.3f -> %.3f (Improvement: %.3f)\n\n", ...
            best_desc, gaussian_current, best_noise, gaussian_current - best_noise);

        I_out = best_candidate;
        gaussian_current = best_noise;
    end

    if gaussian_current <= gaussian_threshold
        fprintf("  >> Gaussian threshold (%.2f) reached after %d iterations!\n", ...
            gaussian_threshold, nlm_iteration);
    elseif nlm_iteration >= max_iterations
        fprintf("  >> Max NLM iterations (%d) reached. Final Gaussian: %.3f\n", ...
            max_iterations, gaussian_current);
    else
        fprintf("  >> Early stopped at iteration %d. Final Gaussian: %.3f\n", ...
            nlm_iteration, gaussian_current);
    end

    final_noise = gaussian_current;
    total_iterations = nlm_iteration;
end
