# ProtoGCN

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2411.18941) [![models](https://img.shields.io/badge/Link-Models-87CEEB.svg)](https://drive.google.com/drive/folders/1BLtlGlv19nY6QcYsVyOBo7nBr3iw5cFl?usp=sharing) [![video](https://img.shields.io/badge/License-MIT-yellow?style=flat)](/LICENSE) [![Hugging Face](https://img.shields.io/badge/Page-Hugging_Face-6C3483?style=flat)](https://huggingface.co/firework8/ProtoGCN)

This is the official PyTorch implementation for "[Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Revealing_Key_Details_to_See_Differences_A_Novel_Prototypical_Perspective_CVPR_2025_paper.html)". The paper is accepted to CVPR 2025.

### Abstract
> In skeleton-based action recognition, a key challenge is distinguishing between actions with similar trajectories of joints due to the lack of image-level details in skeletal representations. Recognizing that the differentiation of similar actions relies on subtle motion details in specific body parts, we direct our approach to focus on the fine-grained motion of local skeleton components. To this end, we introduce ProtoGCN, a Graph Convolutional Network (GCN)-based model that breaks down the dynamics of entire skeleton sequences into a combination of learnable prototypes representing core motion patterns of action units. By contrasting the reconstruction of prototypes, ProtoGCN can effectively identify and enhance the discriminative representation of similar actions. Without bells and whistles, ProtoGCN achieves state-of-the-art performance on multiple benchmark datasets, including NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton, and FineGYM, which demonstrates the effectiveness of the proposed method.

## Installation

```shell
git clone https://github.com/firework8/ProtoGCN.git
cd ProtoGCN
conda env create -f protogcn.yaml
conda activate protogcn
pip install -e .
```

## Data Preparation

PYSKL provides links to the pre-processed skeleton pickle annotations.

- NTU RGB+D: [NTU RGB+D Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)
- NTU RGB+D 120: [NTU RGB+D 120 Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)
- Kinetics-Skeleton: [Kinetics-Skeleton Download Link](https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl)
- FineGYM: [FineGYM Download Link](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)


For Kinetics-Skeleton, since the skeleton annotations are large, please use the [Kinetics Annotation Link](https://www.dropbox.com/scl/fi/5phx0m7bok6jkphm724zc/kpfiles.zip?rlkey=sz26ljvlxb6gwqj5m9jvynpg8&st=47vcw2xb&dl=0) to download the `kpfiles` and extract it under `$ProtoGCN/data/k400` for Kinetics-Skeleton. 

Note that the `kpfiles` needs to be extracted under `Linux`. Additionally, Kinetics-Skeleton requires the dependency `Memcached` to run, which could be referred to [here](https://www.runoob.com/memcached/memcached-install.html). 

You could check the official [Data Doc](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) of PYSKL for more detailed instructions.

## Training & Testing

Please change the config file depending on what you want. You could use the following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.

```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# For example: train on NTU RGB+D X-Sub (Joint Modality) with 1 GPU, with validation, and test the checkpoint.
bash tools/dist_train.sh configs/ntu60_xsub/j.py 1 --validate --test-last --test-best
```

```shell
# Testing
bash tools/dist_test.sh {config_name} {checkpoint_file} {num_gpus} {other_options}
# For example: test on NTU RGB+D X-Sub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ntu60_xsub/j.py checkpoints/CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

```shell
# Ensemble the results
cd tools
python ensemble.py
```

## Pretrained Models

All the checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1BLtlGlv19nY6QcYsVyOBo7nBr3iw5cFl?usp=sharing).

For the detailed performance of pretrained models, please go to the [Model Doc](/data/README.md).

## Acknowledgements

This repo is mainly based on [PYSKL](https://github.com/kennymckormick/pyskl). We also refer to [MS-G3D](https://github.com/kenziyuliu/ms-g3d), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [FR-Head](https://github.com/zhysora/FR-Head).

Thanks to the original authors for their excellent work!

## Citation

If you find ProtoGCN useful in your research, please consider citing our paper:

```
@inproceedings{liu2025revealing,
  title={Revealing key details to see differences: A novel prototypical perspective for skeleton-based action recognition},
  author={Liu, Hongda and Liu, Yunfan and Ren, Min and Wang, Hao and Wang, Yunlong and Sun, Zhenan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29248--29257},
  year={2025}
}
```

## Contact

For any questions, feel free to contact: `hongda.liu@cripac.ia.ac.cn`
