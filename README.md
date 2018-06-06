# Adversarial Training Methods for Semi-Supervised Text Classification
Code for [*Adversarial Training Methods for Semi-Supervised Text Classification*](https://arxiv.org/abs/1605.07725)

This code reproduce the [[Miyato et al., 2017]](https://arxiv.org/abs/1605.07725) with [Chainer](https://github.com/chainer/chainer).


## Setup envirment
Please install [Chainer](https://github.com/chainer/chainer) and [Cupy](https://github.com/cupy/cupy).

You can set up the environment easily with this [*Setup.md*](https://github.com/aonotas/adversarial_text/blob/master/Setup.md).

## Download Pretrain Model
Please download pre-trained model.
```
$ wget http://sato-motoki.com/research/vat/imdb_pretrained_lm.model
```

# Result
Model                                                                           | Error Rate
------------------------------------------------------------------------------- | :---:
Baseline [[Miyato et al., 2017]](https://arxiv.org/pdf/1605.07725.pdf)          | 7.39
Baseline (Our code)                                                             | 6.62
Adversarial [[Miyato et al., 2017]](https://arxiv.org/pdf/1605.07725.pdf)       | 6.21
Adversarial Training (Our code)                                                 | 6.35
Virtual Adversarial Training [[Tensorflow code]](https://github.com/tensorflow/models/tree/master/research/adversarial_text) | 6.40
Virtual Adversarial Training [[Miyato et al., 2017]](https://arxiv.org/pdf/1605.07725.pdf) | 5.91
Virtual Adversarial Training (Our code)                                         | 5.82


# Run
## Pretrain
```
$ python -u pretrain.py -g 0 --layer 1 --dataset imdb --bproplen 100 --batchsize 32 --out results_imdb_adaptive --adaptive-softmax
```
Note that this command takes about 30 hours with single GPU.

## Train (VAT: Semi-supervised setting)
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_vat --lower=0 --use_adv=0 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1
```
Note that this command takes about 8 hours with single GPU.

## Train (Adversarial Training: Supervised setting)
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_adv --lower=0 --use_adv=1 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm.model --use_exp_decay=1 --clip=5.0
```
Note that this command takes about 6 hours with single GPU.

# Authors
We thank Takeru Miyato ([@takerum](https://github.com/takerum)) who suggested that we reproduce the result of a [Miyato et al., 2017].
- Code author: [@aonotas](https://github.com/aonotas/)
- Thanks for Adaptive Softmax implementation: [@soskek](https://github.com/soskek/)
Adaptive Softmax: https://github.com/soskek/efficient_softmax
# Reference
```
[Miyato et al., 2017]: Takeru Miyato, Andrew M. Dai and Ian Goodfellow
Adversarial Training Methods for Semi-Supervised Text Classification.
International Conference on Learning Representation (ICLR), 2017
```
