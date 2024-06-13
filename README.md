# VeCAF: VLM-empowered Collaborative Active Finetuning with Training Objective Awareness
## Abstract

Finetuning a pretrained vision model (PVM) is a common technique for learning downstream vision tasks. The conventional finetuning process with the randomly sampled data points results in diminished training efficiency. To address this drawback, we propose a novel approach, VLM-empowered Collaborative Active Finetuning (VeCAF). VeCAF optimizes a parametric data selection model by incorporating the training objective of the model being tuned. Effectively, this guides the PVM towards the performance goal with improved data and computational efficiency. As vision-language models (VLMs) have achieved significant advancements by establishing a robust connection between image and language domains, we exploit the inherent semantic richness of the text embedding space and utilize text embedding of pretrained VLM models to augment PVM image features for better data selection and finetuning. Furthermore, the flexibility of text-domain augmentation gives VeCAF a unique ability to handle out-of-distribution scenarios without external augmented data. Extensive experiments show the leading performance and high efficiency of VeCAF that is superior to baselines in both in-distribution and out-of-distribution image classification tasks. On ImageNet, VeCAF needs up to 3.3× less training batches to reach the target performance and high efficiency of VeCAF that is superior to baselines in both in-distribution and out-of-distribution image classification tasks.  On ImageNet, VeCAF needs up to 3.3x less training batches to reach the target performance compared to full finetuning and achieves 2.8% accuracy improvement over SOTA methods with the same number of batches. 

[[paper link]](https://arxiv.org/abs/2303.14382)

![overview](overview.png)

## Installation

#### Environment

This codebase has been developed with CUDA 11.2, python 3.7, PyTorch 2.0.1+cu117, and torchvision 0.15.2+cu117. Please install [PyTorch](https://pytorch.org/) according to the instruction on the official website.

You also need to install [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models) for model finetuning with [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md).

You also need to install the environment and code of [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory) under ./sample_tools/ folder and [CLIP](https://github.com/openai/CLIP) for LLM and VLM interaction.

```
pip install timm==0.9.6
```

#### Data Preparation

For [ImageNet](https://www.image-net.org/), you have to manually download it and link to `data_selection/data/ImageNet` and `deit/data/ImageNet`. 

## VeCAF Data Selection

#### Feature Extraction

Before data selection, you need to extract the features with a pretrained model. 

```
cd data_selection/
python extract_feature.py --dataset ${DATASET (ImageNet)}
```

Our default setting applies the DeiT-Small model pretrained with DINO ([ckpt](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth)). You can also specify other models in `data_selection/extract_feature.py`.

## Extreact loss of a pre-trained vision model

Before fine-tuning, you need to get the boundary decision information of the vision language model, as addition information for the data selection.

python -m torch.distributed.launch --nproc_per_node=2 --master_port ${seed} --use_env ${deit_dir}/eval.py \
    --clip-grad 2.0 \
    --eval_interval 50 \
    --data-set ${DATASET (IMNETSUBSET)} \
    --resume ${PATH to the pre-trained vision model} \
    --epochs 1 \
    --output_dir ${PATH to store the output}



#### Data Selection

With extracted features, you can select a subset from the dataset with the following command and put single_turn_mmnew.py under ./data_selection/sample_tools/LLaMA2-accessory/demos/ folder for the single call of LLaMA2-accessory.

```

# For ImageNet:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 --master_port=29528 ./VeCAF/data_selection/sample_tools/VeCAF_ImageNet.py  --feature_path ${PATH to the extracted feature} --percent ${sampling percentage} --weight_dir ${PATH to the loss.pt} --${normalize/exp/log/sigmoid} True --loop ${number of loop}
```


## Model Finetuning

We implement the model finetuning with our selected data subset based on the code base of [deit](https://github.com/facebookresearch/deit). You modify their code to allow the training on the selected subsets.

First, make sure you have downloaded the pretrained ViT model. In our default setting, we finetune the DeiT-Small model pretrained with DINO ([ckpt](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth)).

Then, you can run the following command to finetune the model.

```
cd deit/

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py --clip-grad 2.0 --eval_interval 50 --data-set ${DATASET (IMNETSUBSET)} --subset_ids ${JSON file for selected subset} --resume ${checkpoint (.pth) to be finetuned} --output_dir ${OUTPUT DIR}
```

Example:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 --use_env main.py --clip-grad 2.0 --eval_interval 50 --data-set IMNETSUBSET --subset_ids /data/liuyijiang/zhangrongyu/VeCAF/data_selection/features/ImageNet_dino_base_train_VeCAF_euclidean_temp_0.07_lr_0.001000_scheduler_none_iter_100_sampleNum_12811_ensemble_boundary.json --resume /data/liuyijiang/zhangrongyu/VeCAF/data_selection/pretrained_model/dino_vitbase16_pretrain.pth --output_dir /data/liuyijiang/zhangrongyu/VeCAF/output
```

The finetuning process may be very sensitive to batch size or learning rate. To reproduce the number in the paper, we strongly recommend you to use the above command with 2 GPUs. If you prefer to finetune on a single GPU, you may need to double the batch size or half the learning rate.

## Acknowledgment

The code of this repo is developed based on [dino](https://github.com/facebookresearch/dino) and [deit](https://github.com/facebookresearch/deit). We sincerely thank the authors for making their projects open-source.

## Reference

If you find our work useful, please consider citing the following paper:

```
@article{zhang2024vecaf,
  title={VeCAF: VLM-empowered Collaborative Active Finetuning with Training Objective Awareness},
  author={Zhang, Rongyu and Cai, Zefan and Yang, Huanrui and Liu, Zidong and Gudovskiy, Denis and Okuno, Tomoyuki and Nakata, Yohei and Keutzer, Kurt and Chang, Baobao and Du, Yuan and others},
  journal={arXiv preprint arXiv:2401.07853},
  year={2024}
}
```