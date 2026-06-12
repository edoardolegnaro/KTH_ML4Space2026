# KTH ML4Space 2026

Teaching notebook using DL for solar active-region cutouts.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edoardolegnaro/KTH_ML4Space2026/blob/main/Solar_AR_cutouts.ipynb)

## Files

- `Solar_AR_cutouts.ipynb`: EDA and baseline ML models.
- `slides/`: lecture slides.
- `utils.py`: notebook helper functions for loading, preprocessing, training, and evaluation.

## Dataset

The notebook expects the toy dataset folder, which you can download [here](https://drive.google.com/file/d/1lvzSZblxuszFTdGSucQhKatBXnfQMryi/view?usp=sharing):

```text
arccnet-ar-classification-toy-v20251016/
├── region_classification.parq
└── fits/
```

In Colab, the setup cell downloads `utils.py` from this repository if Colab opened only the notebook. The EDA configuration cell can then download the toy dataset from `DATASET_GOOGLE_DRIVE_URL`; leave the provided public Drive URL in place, or replace it with another zipped/tarred dataset archive or public Drive folder.
