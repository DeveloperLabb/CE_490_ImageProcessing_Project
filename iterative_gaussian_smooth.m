function [I_out, final_edge, final_noise, total_iterations] = iterative_gaussian_smooth(I_in, max_iterations, sigma_list, fs_list)
% Iterative Gaussian smoothing with edge preservation.

    fprintf("\n--- STEP 5: Iterative Gaussian Smoothing ---\n");

    I_out = I_in;

    [edge_initial, ~] = edge_noise_metrics(I_out);
    [edge_current, noise_current] = edge_noise_metrics(I_out);
    step_iteration = 0;
    total_edge_loss = 0;

    fprintf("Initial Edge: %.6f | Noise: %.6f\n", edge_initial, noise_current);
    fprintf("Rule: noise_red/edge_loss max, stop if total edge loss > 5%%\n\n");

    while step_iteration < max_iterations
        step_iteration = step_iteration + 1;

        fprintf("  [Iteration %d] - Edge: %.6f | Noise: %.6f | Total Edge Loss: %.2f%%\n", ...
            step_iteration, edge_current, noise_current, total_edge_loss*100);
        fprintf("  Grid Search başlatılıyor...\n");

        [edge0, noise0] = edge_noise_metrics(I_out);

        best_ratio = -inf;
        best_noise = inf;
        best_desc  = "";
        best_candidate = I_out;
        found_valid = false;

        for sigma = sigma_list
            for fs = fs_list
                candidate = imgaussfilt(I_out, sigma, ...
                    "FilterSize", fs, ...
                    "Padding", "symmetric");

                [edge1, noise1] = edge_noise_metrics(candidate);

                edge_loss       = (edge0 - edge1) / (edge0 + 1e-12);
                noise_reduction = (noise0 - noise1) / (noise0 + 1e-12);

                potential_total_loss = (edge_initial - edge1) / (edge_initial + 1e-12);

                if edge_loss > 0.001
                    ratio = noise_reduction / edge_loss;
                else
                    ratio = noise_reduction * 100;
                end

                fprintf("    sigma=%.2f FS=%d | edge_loss=%.4f noise_red=%.4f ratio=%.2f total=%.2f%%\n", ...
                    sigma, fs, edge_loss, noise_reduction, ratio, potential_total_loss*100);

                if potential_total_loss < 0.1 && noise_reduction > 0 && ratio > best_ratio
                    best_ratio = ratio;
                    best_noise = noise1;
                    best_desc  = sprintf("sigma=%.2f,FS=%d", sigma, fs);
                    best_candidate = candidate;
                    found_valid = true;
                end
            end
        end

        if ~found_valid || best_noise >= noise_current
            fprintf("\n    >>> Stop: no valid parameter (noise reduction + edge loss < 5%%)\n");
            fprintf("    >>> Early stopping.\n\n");
            break;
        end

        [edge_new, noise_new] = edge_noise_metrics(best_candidate);
        total_edge_loss = (edge_initial - edge_new) / (edge_initial + 1e-12);

        fprintf("\n    >>> Best: %s | Ratio: %.2f\n", best_desc, best_ratio);
        fprintf("    >>> Noise: %.6f -> %.6f\n", noise_current, noise_new);
        fprintf("    >>> Total Edge Loss: %.2f%%\n\n", total_edge_loss*100);

        I_out = best_candidate;
        edge_current = edge_new;
        noise_current = noise_new;
    end

    if step_iteration >= max_iterations
        fprintf("  >> Max iterations (%d) reached.\n", max_iterations);
    else
        fprintf("  >> Early stopped at iteration %d.\n", step_iteration);
    end

    final_edge = edge_current;
    final_noise = noise_current;
    total_iterations = step_iteration;
end
