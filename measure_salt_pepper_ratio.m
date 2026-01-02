function ratio = measure_salt_pepper_ratio(img)
% Returns ratio of salt (255) and pepper (0) pixels to total pixels.

    if ~isa(img, 'double')
        img = double(img);
    end
    
    total_pixels = numel(img);
    salt_count = sum(img(:) == 255);
    pepper_count = sum(img(:) == 0);
    noise_pixels = salt_count + pepper_count;
    
    ratio = noise_pixels / total_pixels;
end
