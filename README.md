# VQ-WAE: [Vector Quantized Wasserstein Auto-Encoder](https://arxiv.org/abs/2302.05917)

## Training
The training of a model can be done by calling main.py with the corresponding yaml file. The list of yaml files can be found below.
Please refer to main.py (or execute 'python main.py --help') for the usage of extra arguments.

### Setup steps before training of a model
* Set the checkpoint path "_C.path" (/configs/defaults.py:4) 
* Set the dataset path, "_c.path_dataset" (/configs/defaults.py:5).


### Train a model
Example 1: Gaussian SQ-VAE (I) on CIFAR10
```
python main.py -c "cifar10_sqvae_C512.yaml" --save
```
Example 3: VQ-WAE on CIFAR10
```
python main.py -c "cifar10_vqwae_C512.yaml" --save --dbg --gpu 3
```
python main.py -c "mnist_vqwae_C512.yaml" --save --dbg --gpu 2

python main.py -c "svhn_vqwae_C512.yaml" --save --dbg --gpu 3

### Train a model with dual form WS
python main.py -c "cifar10_vqwae_C512_dual_form.yaml" --save --dbg --gpu 3

### Test a trained model
Example 1: VQ-WAE on CIFAR10
```
python main.py -c "cifar10_vqwae_C512.yaml" --save -timestamp resnet_seed0_0916_0610
```



### Where to find the checkpoints
If the trainning is successful, checkpoint folders will be generated under the folder (cfgs represents the yaml file specified when calling main.py):
```
configs.defaults._C.path + '/' + cfgs.path_spcific
```

**Evaluation:** goto WQVAE/evaluations/ and run:

Fid score:
```
python3 fid_score.py folder_groudtruth_images  folder_recontructed_images --batch-size 192 --gpu 1
```

Lips, PSNR, SSIM score:
```
python evaluation.py --gt_path folder_groudtruth_images  --g_path folder_recontructed_images 
```

**Train generation model:** goto WQVAE/ and run:

```
python3 pixelcnn/gated_pixelcnn.py --dataset LATENT_BLOCK --batch_size 128 --model_name vqvae_OR_MNIST_512_0.25_50000.pth  --epochs 100 --is_label True --device 0
```
*--is_label*: conditional sampling or unconditional sampling

**Image sampling:** goto WQVAE/ and run:

```
python3 generation.py --model_name='vqvae_OR_SVHN_512_0.25_50000.pth' --batch_size 8 --device 0 --is_label 1
```

*--is_label*: conditional sampling or unconditional sampling

**Note:** Other hyper-parameters can be found in Arguments.py

### Running experiments on FFHQ or large datasets (e.g. ImageNet):

We adapt code from https://github.com/CompVis/taming-transformers, particularly:

Please replace **quantize.py** in **/taming-transfromers/taming/modules/vqvae/quantize.py** by our provided **taming/quantize.py** and run experiment following instructions from taming-transformers repo.


## Experiments
"[checkpoint_foldername_with_timestep]" means the folder names under the path "[configs.defaults._C.path + '/' + cfgs.path_spcific]".
These folder names are consist of the model names, the seed indices and the timestamps.

## Dependencies
numpy
scipy
torch
torchvision
PIL
ot

## Acknowledgements
Codes are adapted from https://github.com/sony/sqvae/tree/main/vision. We thank them for their excellent projects.
