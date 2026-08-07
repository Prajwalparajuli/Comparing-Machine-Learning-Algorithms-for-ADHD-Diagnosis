# Data

## Source
ADHD-200 dataset (http://fcon_1000.projects.nitrc.org/indi/adhd200/)  
Athena Pipeline preprocessed data from [NITRC NeuroBureau](https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline)

## Processed Files
- `processed/X.npy` — Feature matrix, shape [620, 868], dtype float64
- `processed/y.npy` — Labels (0=Control, 1=ADHD), shape [620], dtype int32
- `processed/metadata.csv` — Subject IDs, site labels, and original/binarized DX
- `processed/data_summary.json` — Summary statistics

### Feature Composition (868 total)
| Feature Block | Count | Description |
|---|---|---|
| FC PCA | 488 | PCA-reduced functional connectivity (95% variance from 17,955 raw FC features) |
| ReHo | 190 | Regional Homogeneity approximation per CC200 ROI |
| fALFF | 190 | Fractional Amplitude of Low-Frequency Fluctuations (0.01–0.08 Hz) per ROI |

### Class Distribution
| Class | Count | Percentage |
|---|---|---|
| Control (y=0) | 340 | 54.8% |
| ADHD (y=1) | 280 | 45.2% |

### Site Distribution
| Site | Subjects |
|---|---|
| KKI | 83 |
| NYU | 216 |
| NeuroIMAGE | 48 |
| OHSU | 79 |
| Peking_1 | 85 |
| Peking_2 | 67 |
| Peking_3 | 42 |

## Preprocessing Pipeline

### 1. Raw Input
- Format: `.1D` files (tab-separated time-series), one per subject
- Atlas: CC200 (Craddock 200 ROI parcellation, 190 ROIs after removing 2 non-brain columns)
- Source: ADHD-200 Athena pipeline preprocessed data

### 2. Functional Connectivity (FC)
- Per subject: read `.1D` file → drop first 2 columns → `np.corrcoef(rowvar=False)` → 190×190 matrix
- Upper triangle flattened → 17,955 features per subject
- 7 sites merged via `pd.concat()` → 620 subjects × 17,955 features

### 3. PCA Dimensionality Reduction
- `StandardScaler().fit_transform()` on FC features
- `PCA(n_components=488)` — 95% cumulative variance threshold
- **NOTE**: In the original notebooks, PCA was fit on the entire dataset (data leakage). 
  The new pipeline fits PCA only on training folds within cross-validation.

### 4. ReHo (Regional Homogeneity)
- Temporal variance ratio approximation per ROI
- 190 features per subject

### 5. fALFF (Fractional Amplitude of Low-Frequency Fluctuations)
- FFT of each ROI time-series
- Power in 0.01–0.08 Hz band / total power
- 190 features per subject

### 6. Label Binarization
- Original DX: {0=Control, 1=ADHD-Combined, 2=ADHD-Hyperactive-Impulsive, 3=ADHD-Inattentive}
- Binarized: 0=Control, 1=ADHD (any subtype)

## Raw Data
- `raw/` contains the intermediate CSVs produced by the feature extraction notebooks
- Raw ADHD-200 NIfTI/1D volumes are NOT included (requires Data Usage Agreement)
- The processed `X.npy` and `y.npy` files are the starting point for the ML comparison pipeline
