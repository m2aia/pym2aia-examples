import numpy as np
import torch
from torch.utils.data import Dataset

import m2aia as m2

# adaptions to gaussian noise to work on single channeled images
def gaussian_noise(pix, mean=0, sigmas=(0.001, 0.2)):
    sigma = np.random.uniform(sigmas[0], sigmas[1])   # randomize magnitude
    pix = pix * 255
    # adaptively tune the magnitude, hardcode according to the data distribution. every img is [0, 255]
    if pix[pix > 25].shape[0] > 0:       # 1st thre 25
        aver = torch.mean(pix[pix > 25])
    elif pix[pix > 20].shape[0] > 0:     # 2nd thre 20
        aver = torch.mean(pix[pix > 20])
    elif pix[pix > 15].shape[0] > 0:     # 3nd thre 15
        aver = torch.mean(pix[pix > 15])
    elif pix[pix > 10].shape[0] > 0:     # 4nd thre 10
        aver = torch.mean(pix[pix > 10])
    else:
        aver = torch.mean(pix)
        
    sigma_adp = sigma/153*aver           # 153, 0 homogeneous img average pixel intensity

    # scale gray img to [0, 1]
    pix = pix / 255
    # generate gaussian noise
    noise = np.random.normal(mean, sigma_adp, pix.shape)
    # generate image with gaussian noise
    pix_out = pix + torch.tensor(noise)
    pix_out = np.clip(pix_out, 0, 1)
    img_out = pix_out # convert to PIL image
    return torch.as_tensor(img_out, dtype=torch.float32)


class AugmentedDataset(Dataset):
    ''' Torch Dataset using a pyM2aia Dataset to generate data.
        Parameters:
            dataset: m2.BaseDataSet
            transform: transformation function returning an augmented element of 'dataset'
    '''
    def __init__(self, dataset: m2.BaseDataSet, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        I = self.dataset[index]
        X = self.transform(I)
        Y = self.transform(I)
        return X, Y