import segmentation_models_pytorch as sm

backbone = sm.Unet('vgg19', encoder_weights='imagenet', activation='sigmoid')
