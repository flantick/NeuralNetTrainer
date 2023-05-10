import torch
import torchvision
import numpy as np
import random
from torchvision.io import read_image
import cv2
from scipy import ndimage


class BinarySegmentationDataset(torch.utils.data.IterableDataset):
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
    >>> train_dataset = SematicSegmentationDataset(train_df, normalize_func, color_map, with_target=True, shuffle=True)
    >>> train_loader = DataLoader(train_dataset, batch_size=batch_size)
"""

    def __init__(self, dataframe, normalize_func, with_target=True, shuffle=False):
        super().__init__()
        self.dataframe = dataframe
        self.with_target = with_target
        self.shuffle = shuffle
        self.normalize = normalize_func

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):

        list_of_dataframe_indexes = [i for i in range(len(self.dataframe))]
        if self.shuffle:
            random.shuffle(list_of_dataframe_indexes)

        for i in list_of_dataframe_indexes:

            img_name, mask_rle = self.dataframe.iloc[i]

            img = self.readimg(img_name)

            if self.with_target:
                mask = self.readmask(mask_rle, (img.shape[2], img.shape[1], 1))
            else:
                mask = torch.zeros(1)

            yield img, mask

    def readimg(self, img_name):
        img = read_image(img_name, mode=torchvision.io.image.ImageReadMode.RGB)
        img = img.float()
        img =img /255.
        img = self.normalize(img)
        return img

    def readmask(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        mask = img.reshape(shape)
        mask = ndimage.rotate(mask, -90)
        mask = cv2.flip(mask, 1)
        mask = torch.from_numpy(mask)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)

        return mask