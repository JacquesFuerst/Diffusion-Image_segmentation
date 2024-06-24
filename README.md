# Diffusion model for Image segmentation

Denoising Diffusion Probabilistic Models (DDPMs) mark a significant advancement in the development of the field of Computer Vision. Given their success in image generation, the aim of this project was to utilize the feature maps created by a trained DDPM as an insertion into a pixel classifier to perform image segmentation. Furthermore, the impact of different architectural add-ons on the DDPM, which were introduced in the Deep Learning community over the past years, was evaluated. The archtecture and approach was introduced in 'Label-efficient semantic segmentation with diffusion models' [1], which was the paper aimed to be replicated in this project.

## File overview

### diffusion_model_training.ipynb

The training loop for the diffusion model of which the feature maps are extracted to train the pixel classifier.

### PixelClassification.ipynb

The architecture of the pixel classifiier and its training with the feature maps that are extracted from the diffusion model.

### evaluation_diffusion.ipynb

Evaluation of the diffusion model with the FID score metric.

### unet.py

The unet architecture and diffusion process functions for the diffusion model. 

### dataset_utils.py

Useful functions for processing the dataset one wants to train the diffusion model on, built to work for torchvision datasets.



## References

[1] Dmitry Baranchuk et al. “Label-efficient semantic segmentation with diffusion models”. In: arXiv preprint arXiv:2112.03126 (2021).
