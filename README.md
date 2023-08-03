# MAPLE: Semi-Supervised Learning with Multi-Alignment and Pseudo-Learning

This project hosts the code for implementing the MAPLE algorithm for Semi-Supervised Learning with Multi-Alignment and Pseudo-Learning. 

The full paper is available at: (soon to come).

Implementation based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

# How to run

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer MAPLE \
--source-domains amazon \
--target-domains webcam \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only/office31.yaml \
--output-dir output/source_only_office31_test \
--eval-only \
--model-dir output/source_only_office31 \
--load-epoch 20
```

Note that `--model-dir` takes as input the directory path which was specified in `--output-dir` in the training stage.


