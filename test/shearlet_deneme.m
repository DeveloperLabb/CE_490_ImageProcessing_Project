addpath(genpath("C:\Users\husey\OneDrive\Desktop\CE_490\SA_DCT"));

ad_im = adaptive_median_filtering(imread("C:\Users\husey\OneDrive\Desktop\CE_490\CE_490_ImageProcessing_Project\project_images\degraded\degraded_boat.png"), 5);

function_shape_explorer(double(ad_im));