# KTH ML4Space 2026

Teaching notebook for the solar active-region cutouts.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edoardolegnaro/KTH_ML4Space2026/blob/main/Solar_AR_cutouts.ipynb)

## Files

- `Solar_AR_cutouts.ipynb`: EDA and baseline ML models.
- `utils.py`: notebook helper functions for loading, preprocessing, training, and evaluation.

## Dataset

The notebook expects the toy dataset folder:

```text
arccnet-ar-classification-toy-v20251016/
├── region_classification.parq
└── fits/
```

In Colab, paste the public Google Drive dataset URL into `DATASET_GOOGLE_DRIVE_URL` in the EDA configuration cell. The notebook can download either a zipped/tarred dataset archive or a public Drive folder.
