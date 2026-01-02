function [I_out, final_edge, final_noise, total_iterations] = iterative_freq_sharpen(I_in, max_iterations, k_list, cutoff_list, ratio_threshold, noise_threshold)
% Iterative frequency-domain sharpening with edge/noise ratio control.

    fprintf("\n--- STEP 3: Iterative Freq Sharpen (Edge > Noise Gain) ---\n");

    I_out = I_in;

    [edge_current, noise_current] = edge_noise_metrics(I_out);
    sharpen_iteration = 0;

    fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_current, noise_current);
    fprintf("Rule: Continue while edge_gain > noise_gain\n\n");

    while sharpen_iteration < max_iterations
        sharpen_iteration = sharpen_iteration + 1;

        fprintf("  [Sharpen Iteration %d] - Edge: %.6f | Noise: %.6f\n", ...
            sharpen_iteration, edge_current, noise_current);
        fprintf("  Grid Search başlatılıyor...\n");

        [edge0, noise0] = edge_noise_metrics(I_out);

        best_ratio      = inf;
        best_edge_gain  = 0;
        best_noise_gain = 0;
        best_desc       = "";
        best_candidate  = I_out;

        for k = k_list
            for c = cutoff_list
                candidate = freq_highboost_gauss(I_out, k, c);

                [edge1, noise1] = edge_noise_metrics(candidate);

                edge_norm  = edge1  / (edge0  + 1e-12);
                noise_norm = noise1 / (noise0 + 1e-12);

                edge_gain  = edge_norm  - 1.0;
                noise_gain = noise_norm - 1.0;

                if edge_gain > 0
                    ratio = noise_gain / edge_gain;
                else
                    ratio = inf;
                end

                fprintf("    k=%.2f c=%.3f | edge_gain=%.4f noise_gain=%.4f | ratio=%.4f\n", ...
                    k, c, edge_gain, noise_gain, ratio);

                if ratio < best_ratio
                    best_ratio      = ratio;
                    best_edge_gain  = edge_gain;
                    best_noise_gain = noise_gain;
                    best_desc       = sprintf("k=%.2f,c=%.3f", k, c);
                    best_candidate  = candidate;
                end
            end
        end

        % Check ratio threshold
        if best_ratio >= ratio_threshold
            fprintf("\n    >>> Stop: best ratio = %.4f >= %.1f threshold\n", ...
                best_ratio, ratio_threshold);
            fprintf("    >>> Noise increasing too fast. Early stopping.\n\n");
            break;
        end

        [edge_new, noise_new] = edge_noise_metrics(best_candidate);

        % Check noise threshold
        if noise_new > noise_threshold
            fprintf("\n    >>> Stop: noise = %.6f > %.4f threshold\n", ...
                noise_new, noise_threshold);
            fprintf("    >>> Noise too high. Not applying this iteration.\n\n");
            break;
        end

        fprintf("\n    >>> Best: %s\n", best_desc);
        fprintf("    >>> Edge gain: %.4f | Noise gain: %.4f | Ratio: %.4f\n", ...
            best_edge_gain, best_noise_gain, best_ratio);
        fprintf("    >>> Edge: %.6f -> %.6f | Noise: %.6f -> %.6f\n\n", ...
            edge_current, edge_new, noise_current, noise_new);

        I_out = best_candidate;
        edge_current = edge_new;
        noise_current = noise_new;
    end

    if sharpen_iteration >= max_iterations
        fprintf("  >> Max sharpen iterations (%d) reached.\n", max_iterations);
    else
        fprintf("  >> Early stopped at iteration %d.\n", sharpen_iteration);
    end

    fprintf("\n*** STEP 3 FINAL ***\n");
    fprintf("Total Iterations: %d\n", sharpen_iteration);
    fprintf("Final Edge: %.6f | Noise: %.6f\n\n", edge_current, noise_current);

    final_edge = edge_current;
    final_noise = noise_current;
    total_iterations = sharpen_iteration;
end
