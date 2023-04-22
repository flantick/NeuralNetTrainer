import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
import numpy as np
import random
from torchvision.io import read_image


'''
This class for the semantic_segmentation.ipynb
'''

class ImageSegmentationDataset(torch.utils.data.IterableDataset):
    """
Dataset for image segmentation tasks.

Args:
    dataframe (pandas.DataFrame): Dataframe containing the paths to the images.
    normalize_func (callable): Normalization function to be applied to the images.
    color_map (dict): Dictionary representing color mappings.
    with_target (bool, optional): Whether or not to include the target masks. Default is True.
    shuffle (bool, optional): Whether or not to shuffle the dataset. Default is False.

Attributes:
    dataframe (pandas.DataFrame): Dataframe containing the paths to the images.
    with_target (bool): Whether or not to include the target masks.
    shuffle (bool): Whether or not to shuffle the dataset.
    normalize (callable): Normalization function to be applied to the images.
    color_map (dict): Dictionary representing color mappings.

Methods:
    __len__(): Returns the length of the dataset.
    __iter__(): Iterates over the dataset and yields the images and masks.
    readimg(img_name): Reads the image from the given path and applies normalization.
    readmask(mask_name): Reads the target mask from the given path and converts it to a binary mask.
    pt_rgb_to_mask(img, color_map): Converts an RGB image mask to a binary mask.

Example:
    >>> train_dataset = ImageSegmentationDataset(train_df, normalize_func, color_map, with_target=True, shuffle=True)
    >>> train_loader = DataLoader(train_dataset, batch_size=batch_size)
"""

    def __init__(self, dataframe, normalize_func, color_map, with_target=True, shuffle=False):
        super().__init__()
        self.dataframe = dataframe
        self.with_target = with_target
        self.shuffle = shuffle
        self.normalize = normalize_func
        self.color_map = color_map

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):

        list_of_dataframe_indexes = [i for i in range(len(self.dataframe))]
        if self.shuffle:
            random.shuffle(list_of_dataframe_indexes)

        for i in list_of_dataframe_indexes:

            img_name = self.dataframe.iloc[i][0]

            if self.with_target:
                mask_name = img_name[:16]+'segmentation'+img_name[22:]
                mask = self.readmask(mask_name)
            else:
                mask = torch.zeros(1)

            img = self.readimg(img_name)

            yield img, mask

    def readimg(self, img_name):
        img = read_image(img_name, mode=torchvision.io.image.ImageReadMode.RGB)
        img = img.float()
        img =img /255.
        img = self.normalize(img)
        return img

    def readmask(self, mask_name):
        mask = read_image(mask_name, mode=torchvision.io.image.ImageReadMode.RGB)

        mask = T.ToPILImage()(mask)
        mask_np = np.asarray(mask, dtype='uint8')
        mask_np = ImageSegmentationDataset.pt_rgb_to_mask(mask_np, self.color_map)
        mask = torch.asarray(mask_np)
        mask = mask.float()
        return mask

    @staticmethod
    def pt_rgb_to_mask(img, color_map):
        '''
        Converts a RGB image mask of shape [batch_size, h, w, 3] to Binary Mask of shape [batch_size, classes, h, w]
        Parameters:
            img: A RGB img mask
            color_map: Dictionary representing color mappings
        returns:
            out: A Binary Mask of shape [batch_size, classes, h, w]
        '''
        num_classes = len(color_map)
        shape = (num_classes,)+img.shape[:2]
        out = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(color_map):
            res = np.all(img.reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[1:])
            out[i,:,:] = res
        return out