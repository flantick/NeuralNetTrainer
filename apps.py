import torch
import torchvision
import torchvision.transforms as T
from torchvision.io import read_image
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


class pleaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, gen_df=None):
        super().__init__()
        self.gen_df =gen_df

    def __iter__(self):
        for i in range(len(self.gen_df)):
            img_name, mask_rle = self.gen_df.sample(1).values[0]
            img = read_image(img_name, mode=torchvision.io.image.ImageReadMode.RGB)
            img = img.float()
            img =img /255.
            mask = self.rle_decode(mask_rle)
            mask = ndimage.rotate(mask, -90)
            mask = cv2.flip(mask, 1)
            mask = torch.from_numpy(mask)
            mask = mask.float()

            yield img, torch.unsqueeze(mask, 0)

    def __len__(self)  :  # <-here
        return len(self.gen_df)

    def rle_decode(self, mask_rle, shape=(640, 480, 1)):
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

        img = img.reshape(shape)

        return img


# def prepare():
#   global data_t, data_v
#   data = pd.read_csv('dataset/train/segmentation.csv')
#   data_t, data_v = train_test_split(data, train_size=0.8)


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

