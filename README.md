# SHL Spoken English Grammar Scoring

Machine learning solution for the SHL Research Engineer hiring assessment.

## Problem

The objective is to predict a continuous grammar score from spoken English audio recordings. The provided training data contains labeled audio recordings with grammar scores ranging from 0 to 5.

## Approach

The solution treats the task as a supervised regression problem.

### Audio Feature Extraction

The following acoustic features were extracted from each audio recording:

- 13 MFCC means
- 13 MFCC standard deviations
- RMS energy mean and standard deviation
- Zero-crossing rate mean and standard deviation
- Spectral centroid mean and standard deviation
- Spectral bandwidth mean and standard deviation
- Spectral rolloff mean and standard deviation
- Audio duration

This produces a total of 37 features per recording.

### Model

An `ExtraTreesRegressor` was used with:

- 500 trees
- `min_samples_leaf = 1`
- `random_state = 42`
- Parallel processing enabled

An 80/20 train-validation split was used during model evaluation.

## Results

Validation results:

| Metric | Score |
|---|---:|
| RMSE | 0.6823 |
| Pearson Correlation | 0.8656 |

The final model was then trained using all 769 labeled training recordings and used to generate predictions for the 216 test recordings.

The Kaggle public leaderboard score for the submitted solution was **0.7435**.

## Repository Contents

- `shl-grammar-scoring-final-solution.ipynb` — Final Kaggle notebook containing the complete implementation.

## Technologies

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- SciPy
- Jupyter / Kaggle Notebook

## Note

The original audio dataset is not included in this repository. It is provided through the SHL Kaggle competition environment.
