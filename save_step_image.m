function save_step_image(outDir, name, stepIdx, stepLabel, I)

    if ischar(outDir), outDir = string(outDir); end
    if ischar(name),   name   = string(name);   end
    if ischar(stepLabel), stepLabel = string(stepLabel); end

    if ~exist(outDir, "dir"), mkdir(outDir); end

    imgDir = fullfile(outDir, char(name));
    if ~exist(imgDir, "dir"), mkdir(imgDir); end

    % Convert to uint8 for saving (keeps grayscale/RGB as-is)
    if ~isa(I, "uint8")
        if (isa(I,"double") || isa(I,"single")) && max(I(:)) <= 1.0
            Iu = im2uint8(I);
        else
            Iu = uint8(max(0, min(255, I)));
        end
    else
        Iu = I;
    end

    safeLabel = regexprep(char(stepLabel), '[^\w\-]+', '_'); % safe filename
    outPath   = fullfile(imgDir, sprintf("%02d_%s_%s.png", stepIdx, safeLabel, char(name)));

    imwrite(Iu, outPath);
end
