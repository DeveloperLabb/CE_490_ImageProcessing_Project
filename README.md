# CE 490 - Image Processing Project

## Multi-Stage Image Restoration Pipeline for Salt-Pepper and Gaussian Noise Removal

This project implements a comprehensive **6-step image restoration pipeline** designed to restore degraded grayscale images affected by **Salt & Pepper noise** and **Gaussian noise**. The pipeline combines spatial-domain and frequency-domain techniques with iterative optimization strategies.

---

##  Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methods Used](#-methods-used)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Output](#-output)
- [Performance](#-performance)
- [Authors](#-authors)
- [License](#-license)

---

##  Overview

The main script `process_pipeline_final.m` processes 5 test images:
- **Boat**
- **Baboon**
- **Barbara**
- **Peppers**
- **Cameraman**

Each image goes through a 6-step restoration pipeline, with metrics (MSE, PSNR, SSIM) computed at each stage.

**Estimated Runtime:** ~1.5 minutes for all 5 images

---

##  Pipeline Architecture

### Step 0: Noise Analysis
- Estimates **Salt & Pepper ratio** using pixel counting (0/255 values)
- Estimates **Gaussian noise standard deviation** using Laplacian-MAD method
- Computes baseline metrics (MSE, PSNR, SSIM) between original and degraded images

### Step 1: Iterative Adaptive Median Filter
- **Purpose:** Remove Salt & Pepper noise
- **Method:** Adaptive Median Filter with variable window size (3×3 to 5×5)
- **Termination:** When S&P ratio drops below threshold (1%) or max iterations reached
- **Function:** `adaptive_median_filtering.m`

### Step 2: Iterative Non-Local Means (NLM) Filter
- **Purpose:** Reduce Gaussian noise while preserving edges
- **Method:** Grid search over NLM parameters (DegreeOfSmoothing, SearchWindowSize, ComparisonWindowSize)
- **Termination:** When Gaussian noise estimate drops below threshold (0.1)
- **Function:** `iterative_nlm_filter.m`

### Step 3: Iterative Frequency-Domain Sharpening
- **Purpose:** Restore lost edges/details
- **Method:** Gaussian high-boost filter in frequency domain
- **Termination:** When noise-to-edge gain ratio exceeds threshold (early stopping)
- **Function:** `iterative_freq_sharpen.m`, `freq_highboost_gauss.m`

### Step 4: Edge-Masked Laplacian Enhancement
- **Purpose:** Further edge enhancement with noise control
- **Method:** 
  - Apply Laplacian filter to sharpened image (Step 3)
  - Apply Wiener filter to denoise edge map
  - Add smoothed edge map back to NLM result (Step 2)
- **Functions:** Laplacian kernel, `wiener2()`, average filter

### Step 5: Iterative Gaussian Smoothing
- **Purpose:** Final noise reduction with edge preservation
- **Method:** Grid search over Gaussian filter parameters (sigma, filter size)
- **Termination:** When edge loss exceeds 10% or no valid improvement found
- **Function:** `iterative_gaussian_smooth.m`

### Step 6: Final NLM Pass
- **Purpose:** Final Gaussian noise cleanup
- **Method:** Same as Step 2 with tighter threshold (0.08)
- **Function:** `iterative_nlm_filter.m`

---

##  Methods Used

### Noise Estimation Functions

| Function | Description |
|----------|-------------|
| `measure_salt_pepper_ratio.m` | Counts pixels with value 0 or 255 to estimate S&P noise ratio |
| `estimate_gaussian_noise.m` | Uses Laplacian-MAD (Median Absolute Deviation) method for blind noise estimation |
| `edge_noise_metrics.m` | Computes edge strength (Sobel-based) and flat-region noise (Laplacian roughness) |

### Filtering Functions

| Function | Description |
|----------|-------------|
| `adaptive_median_filtering.m` | Adaptive median filter with variable window size (3×3 to max) |
| `iterative_nlm_filter.m` | Iterative Non-Local Means with grid search optimization |
| `iterative_gaussian_smooth.m` | Iterative Gaussian smoothing with edge-loss constraint |
| `freq_highboost_gauss.m` | Frequency-domain Gaussian high-boost sharpening |
| `iterative_freq_sharpen.m` | Iterative frequency sharpening with noise-gain ratio control |

### Metric Functions

| Function | Description |
|----------|-------------|
| `compare_original_restored.m` | Computes MSE, PSNR, and SSIM between original and restored images |

### Utility Functions

| Function | Description |
|----------|-------------|
| `save_step_image.m` | Saves intermediate results to output folder |
| `show_compare_images.m` | Displays comparison figures (Original, Input, Output) |

---

##  Project Structure

```
CE_490_ImageProcessing_Project/
│
├── process_pipeline_final.m          # Main pipeline script
├── pre_analysis.m                    # Pre-analysis utilities
│
├── # Noise Estimation
├── measure_salt_pepper_ratio.m       # S&P noise ratio estimation
├── estimate_gaussian_noise.m         # Gaussian noise estimation (Laplacian-MAD)
├── edge_noise_metrics.m              # Edge strength & noise metrics
│
├── # Filtering
├── adaptive_median_filtering.m       # Adaptive median filter
├── iterative_nlm_filter.m            # Iterative Non-Local Means filter
├── iterative_gaussian_smooth.m       # Iterative Gaussian smoothing
├── freq_highboost_gauss.m            # Frequency-domain high-boost filter
├── iterative_freq_sharpen.m          # Iterative frequency sharpening
├── edge_masked_laplacian_enhance.m   # Edge-masked Laplacian enhancement
│
├── # Metrics
├── compare_original_restored.m       # MSE, PSNR, SSIM computation
│
├── # Utilities
├── save_step_image.m                 # Save intermediate images
├── show_compare_images.m             # Display comparison figures
│
├── project_images/
│   ├── clean/                        # Original images
│   └── degraded/                     # Degraded images (with noise)
│
├── outputs/                          # Generated outputs per image
│   ├── baboon/
│   ├── barbara/
│   ├── boat/
│   ├── cameraman/
│   └── peppers/
│
├── presentation/                     # Presentation files
│
└── test/                             # Experimental/test scripts
```

---

##  Requirements

- **MATLAB R2019b or later** (for `imnlmfilt` function)
- **Image Processing Toolbox**
- Required functions:
  - `imnlmfilt` - Non-Local Means filter
  - `imgaussfilt` - Gaussian filter
  - `wiener2` - 2D Wiener filter
  - `immse`, `psnr`, `ssim` - Quality metrics

---

##  Usage

### Running the Complete Pipeline

1. Open MATLAB and navigate to the project directory:
   ```matlab
   cd('path/to/CE_490_ImageProcessing_Project')
   ```

2. Ensure all images are placed in the correct folders:
   - `project_images/clean/original_<name>.png`
   - `project_images/degraded/degraded_<name>.png`

3. Run the main script:
   ```matlab
   process_pipeline_final
   ```

4. The pipeline will:
   - Process all 5 images sequentially
   - Display comparison figures at each step
   - Print detailed metrics to the console
   - Save intermediate results to `outputs/<image_name>/`

### Customizing the Pipeline

You can modify the following parameters in `process_pipeline_final.m`:

```matlab
% Step 1: Adaptive Median
sp_threshold = 0.01;       % Target S&P ratio
max_iterations = 10;        % Max iterations

% Step 2 & 6: NLM
degree_list = [5 10 15 20 30];  % DegreeOfSmoothing values
sw_list = [11 21];              % SearchWindowSize values
cw_list = [3 5 7];              % ComparisonWindowSize values

% Step 3: Frequency Sharpening
k_list = [0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.8 2.0];  % Boost factors
cutoff_list = [0.03 0.05 0.07 0.09 0.11 0.13];    % Cutoff frequencies

% Step 5: Gaussian Smoothing
sigma_list = [0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0];
fs_list = [3 5 7 9 11 13];  % Filter sizes
```

---

##  Output

### Console Output
- Noise estimation results (estimated vs true values)
- Per-iteration progress for each step
- Summary table with MSE, PSNR, SSIM at each step
- Global average metrics across all images

### Saved Images
Each image generates the following outputs in `outputs/<image_name>/`:

| File | Description |
|------|-------------|
| `00_original_<name>.png` | Original clean image |
| `01_degraded_<name>.png` | Input degraded image |
| `02_step1_adaptive_median_<name>.png` | After S&P removal |
| `03_step2_nlm_<name>.png` | After first NLM pass |
| `04_step3_sharpen_<name>.png` | After frequency sharpening |
| `45_laplace_output_<name>.png` | Laplacian edge map (debug) |
| `05_step4_laplacian_<name>.png` | After Laplacian enhancement |
| `06_step5_gaussian_<name>.png` | After Gaussian smoothing |
| `07_step6_nlm_again_final_<name>.png` | Final restored image |

---

##  Performance

| Metric | Description |
|--------|-------------|
| **Runtime** | ~1.5 minutes for all 5 images |
| **Memory** | Moderate (single image in memory at a time) |

### Results

| Image | MSE Reduction | PSNR Gain (dB) | SSIM Improvement |
|-------|---------------|----------------|------------------|
| Boat | 83.2% | ↑ 7.78 | ↑ 0.475 |
| Baboon | 60.5% | ↑ 4.05 | ↑ 0.210 |
| Barbara | 76.1% | ↑ 6.26 | ↑ 0.472 |
| Peppers | 94.5% | ↑ 12.81 | ↑ 0.722 |
| Cameraman | 75.4% | ↑ 6.07 | ↑ 0.544 |
| **Average** | **77.9%** | **+ 7.39** | **+ 0.485** |

---

##  Authors

* **Hüseyin Yontar**   `HuseyinYontar`
* **Mert Koğuş**   `DeveloperLabb`
 
---

##  License

This project is for educational purposes as part of the CE 490 course.
