================================================================================
           CE 490 - IMAGE PROCESSING PROJECT - HOW TO RUN
================================================================================

REQUIREMENTS
------------
- MATLAB R2019b or later
- Image Processing Toolbox


QUICK START
-----------
1. Open MATLAB

2. Navigate to the project folder:
   >> cd('path/to/CE_490_ImageProcessing_Project')

3. Run the main script:
   >> process_pipeline_final


INPUT FILES
-----------
Place your images in these folders before running:

  project_images/clean/       -> Original (clean) images
      - original_boat.png
      - original_baboon.png
      - original_barbara.png
      - original_peppers.png
      - original_cameraman.png

  project_images/degraded/    -> Degraded (noisy) images
      - degraded_boat.png
      - degraded_baboon.png
      - degraded_barbara.png
      - degraded_peppers.png
      - degraded_cameraman.png


OUTPUT
------
Results are saved in:  outputs/<image_name>/

Each image generates:
  - 00_original_<name>.png          : Original clean image
  - 01_degraded_<name>.png          : Input degraded image
  - 02_step1_adaptive_median_<name>.png  : After S&P noise removal
  - 03_step2_nlm_<name>.png         : After NLM filtering
  - 04_step3_sharpen_<name>.png     : After frequency sharpening
  - 05_step4_laplacian_<name>.png   : After Laplacian enhancement
  - 06_step5_gaussian_<name>.png    : After Gaussian smoothing
  - 07_step6_nlm_again_final_<name>.png  : Final restored image


RUNTIME
-------
Approximately 1.5 minutes for all 5 images.


AUTHORS
-------
Huseyin Yontar - HuseyinYontar
Mert Kogus - DeveloperLabb

================================================================================
