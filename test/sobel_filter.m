function [Gx, Gy, Gmag] = sobel_filter(I_in)
%SOBEL_FILTER Applies Sobel edge detection
%
%   [Gx, Gy, Gmag] = sobel_filter(I_in)
%
%   INPUT:
%       I_in  : grayscale image
%
%   OUTPUT:
%       Gx    : Sobel horizontal edges
%       Gy    : Sobel vertical edges
%       Gmag  : Gradient magnitude (sqrt(Gx^2 + Gy^2))

    I = double(I_in);

    %% Sobel kernels
    Sx = [ -1  0  1;
           -2  0  2;
           -1  0  1 ];

    Sy = [ -1 -2 -1;
            0  0  0;
            1  2  1 ];

    %% Convolution
    Gx = conv2(I, Sx, 'same');
    Gy = conv2(I, Sy, 'same');

    %% Gradient magnitude
    Gmag = sqrt(Gx.^2 + Gy.^2);

    %% Normalize & convert to uint8 for visualization
    Gx = uint8(255 * mat2gray(Gx));
    Gy = uint8(255 * mat2gray(Gy));
    Gmag = uint8(255 * mat2gray(Gmag));
end
