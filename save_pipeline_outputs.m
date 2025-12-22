function save_pipeline_outputs(outDir, name, I_original, I_degraded, I_final)

    if ischar(outDir), outDir = string(outDir); end
    if ischar(name),   name   = string(name);   end

    imgDir = fullfile(outDir, char(name));
    if ~exist(imgDir, "dir")
        mkdir(imgDir);
    end

    p_original = fullfile(imgDir, sprintf("original_%s.png", char(name)));
    p_degraded = fullfile(imgDir, sprintf("degraded_%s.png", char(name)));
    p_final    = fullfile(imgDir, sprintf("final_%s.png",    char(name)));

    imwrite(I_original, p_original);
    imwrite(I_degraded, p_degraded);
    imwrite(I_final,    p_final);

    fprintf("Saved outputs for %s:\n", upper(char(name)));
    fprintf("  %s\n", p_original);
    fprintf("  %s\n", p_degraded);
    fprintf("  %s\n", p_final);
end
