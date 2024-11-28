# Pretrained Models

## Introduction

We present the detailed performance on various datasets and modalities. 

All the checkpoints and training logs are provided in the [Google Drive](https://drive.google.com/drive/folders/1BLtlGlv19nY6QcYsVyOBo7nBr3iw5cFl?usp=sharing). We sincerely hope that this repo could be helpful for your research.

## Experimental Results

The detailed results for pretrained models are displayed below:

| Modality | NTU 60 X-Sub | NTU 60 X-View | NTU 120 X-Sub | NTU 120 X-Set | Kinetics-Skeleton | FineGYM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Joint| [91.54](https://drive.google.com/drive/folders/1uiQoGaDuN2UpXizmYrGQACHmUSLiXcvO?usp=drive_link) | [96.33](https://drive.google.com/drive/folders/1v0DCjo10OO6hQFXQGfO33J_U_iyP_EsV?usp=drive_link) | [85.52](https://drive.google.com/drive/folders/1mI04KbyDFOpf_4jacjjiTaDOcc6m4BVB?usp=drive_link) | [88.35](https://drive.google.com/drive/folders/1XIuHSiWeLP3G273XsotWVwtwbKhX2_uO?usp=drive_link) | [48.02](https://drive.google.com/drive/folders/16NL36IOkIQY3pQMXSpgf8cv_qgh_Oc8A?usp=drive_link) | [93.28](https://drive.google.com/drive/folders/1I0VFWJyRs5ksmRecg38BFC-HCGA7rpRD?usp=drive_link) |
| Bone | [91.98](https://drive.google.com/drive/folders/1nzNz2tGEB8ndjN_bEhFmdvaFSTPcM5UD?usp=drive_link) | [96.15](https://drive.google.com/drive/folders/1ZRGrkeypqWyHFEvX4NwVDb7ClmsBszPi?usp=drive_link) | [88.96](https://drive.google.com/drive/folders/1dTkX5dNxKePiyyc7UGVWyzYlHzO9xfuc?usp=drive_link) | [90.01](https://drive.google.com/drive/folders/1MeuBnNRpdFIpXVIu-29rPJlSnQrbFowi?usp=drive_link) | [47.06](https://drive.google.com/drive/folders/1kH-eXPmvTv8PTDSifkf1bfS5UXShV2CA?usp=drive_link) | [94.84](https://drive.google.com/drive/folders/12S9J8pGrDAR8ctcTHvW48ZpJ1gJqOi5j?usp=drive_link) |
|K-Bone| [91.59](https://drive.google.com/drive/folders/1euYfWszVxhs_MLjtczczaPYFQQ8OokEv?usp=drive_link) | [96.61](https://drive.google.com/drive/folders/1TJROSie3p4OoB7piu7ur4TxcDFzjzQYc?usp=drive_link) | [88.30](https://drive.google.com/drive/folders/1po2sYNKZdAiT05pYX2h8R_kXTHe26Fyk?usp=drive_link) | [89.65](https://drive.google.com/drive/folders/18dUXnwKLSoUDMLaE8TmwaDQWdC1tYuI2?usp=drive_link) | [45.86](https://drive.google.com/drive/folders/1ZyyVaOnOXiGGr6nAryDgOr5kIWaF72KL?usp=drive_link) | [94.44](https://drive.google.com/drive/folders/1b9Fm2S5rC97qQAI-3Zm9SOr-hR7NeKV8?usp=drive_link) |
| 2-ensemble | [92.96](https://drive.google.com/drive/folders/1_5fVMEXQrThQqYUWDErUBhYBaiQzW5Gr?usp=drive_link) | [97.23](https://drive.google.com/drive/folders/1YxmiC_uHfM9Njl07O2JwDIpUnSa6qOrC?usp=drive_link) | [89.75](https://drive.google.com/drive/folders/122H7HQCnXRJsahjSdRBdydccxLXMwj2Y?usp=drive_link) | [91.23](https://drive.google.com/drive/folders/1d1pDhTYWkugwvV8OocE8QGrGDz1ORMu3?usp=drive_link) | [49.85](https://drive.google.com/drive/folders/19-TMsFi9b2BQxTVWuqngEDLUT-bOSPnL?usp=drive_link) | [95.35](https://drive.google.com/drive/folders/1_Tt19hO1Z2IpVy0Tn9rhXLp4QqlRLzbk?usp=drive_link) |
| 4-ensemble | [93.53](https://drive.google.com/drive/folders/1_5fVMEXQrThQqYUWDErUBhYBaiQzW5Gr?usp=drive_link) | [97.49](https://drive.google.com/drive/folders/1YxmiC_uHfM9Njl07O2JwDIpUnSa6qOrC?usp=drive_link) | [90.43](https://drive.google.com/drive/folders/122H7HQCnXRJsahjSdRBdydccxLXMwj2Y?usp=drive_link) | [91.86](https://drive.google.com/drive/folders/1d1pDhTYWkugwvV8OocE8QGrGDz1ORMu3?usp=drive_link) | [51.33](https://drive.google.com/drive/folders/19-TMsFi9b2BQxTVWuqngEDLUT-bOSPnL?usp=drive_link) | [95.62](https://drive.google.com/drive/folders/1_Tt19hO1Z2IpVy0Tn9rhXLp4QqlRLzbk?usp=drive_link) |
| 6-ensemble | [**93.81**](https://drive.google.com/drive/folders/1_5fVMEXQrThQqYUWDErUBhYBaiQzW5Gr?usp=drive_link) | [**97.76**](https://drive.google.com/drive/folders/1YxmiC_uHfM9Njl07O2JwDIpUnSa6qOrC?usp=drive_link) | [**90.92**](https://drive.google.com/drive/folders/122H7HQCnXRJsahjSdRBdydccxLXMwj2Y?usp=drive_link) | [**92.16**](https://drive.google.com/drive/folders/1d1pDhTYWkugwvV8OocE8QGrGDz1ORMu3?usp=drive_link) | [**51.85**](https://drive.google.com/drive/folders/19-TMsFi9b2BQxTVWuqngEDLUT-bOSPnL?usp=drive_link) | [**95.94**](https://drive.google.com/drive/folders/1_Tt19hO1Z2IpVy0Tn9rhXLp4QqlRLzbk?usp=drive_link) |

We adopt the widely-used six-stream ensemble strategy introduced in [InfoGCN](https://github.com/stnoah1/infogcn). Here K-Bone denotes the newly skeleton representation proposed by InfoGCN. Interestingly, we find that the improvement of multi-stream ensemble method mainly comes from *complementarity* and *stochasticity*. For well-performing models, *stochastic boosting of single-modality* is more efficient than *complementary boosting of motion-modality*. The detailed comparisons for various datasets are provided in `{dataset}_ensemble.py`.


In addition, we use two augmentation techniques [Flip](/protogcn/datasets/pipelines/pose_related.py#L186) and [Part Drop](/protogcn/datasets/pipelines/pose_related.py#L220) that could provide performance gains. Notably, due to randomness, they may also make the performance fluctuate. You could choose whether or not to use these augmentations based on the actual needs.