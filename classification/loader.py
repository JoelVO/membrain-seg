import mrcfile
from glob import glob
import numpy as np
import scipy
import torch
import os
from membrain_seg.tomo_preprocessing.matching_utils.px_matching_utils import (determine_input_shape, fourier_cropping,
                                                                              fourier_extend)


class Loader:
    def __init__(self, path: str, crop_size: tuple, pixel_out: float = 10., change_probability: float = 0.05):
        """
        Parameters
        ----------
        path path: where to find tomograms and membranes annotations as well as a file where to store labels.
        crop_size: chosen crop size.
        pixel_out: desired pixel size to standarized training data.
        change_probability: probability to choose another tomogram to extract crops from.
        """
        self.path = path
        self.crop_size = crop_size
        self.pixel_out = pixel_out
        self.change_probability = change_probability

        self.names = glob(f'{path}/czi/*')
        self.pixel_size = np.load(f'{path}/pixel_size.npy', allow_pickle=True).item()

        self.change_tomogram()

    def __call__(self, batch_size: int, device: torch.device = torch.device('cpu'),
                 memory_bank_size: int = 1024, make_memory_bank: bool = False):
        """
        Parameters
        ----------
        batch_size: samples per batch.
        device: device where send the tensors to.
        memory_bank_size: size of memory back used while training.
        make_memory_bank: whether the loader is being used to create a memory bank or to batch.

        Returns
        -------
        torch.tensor
        """
        if make_memory_bank:
            negative_bank = []
            for _ in range(memory_bank_size // 2):
                anchor, positive, negative = self.get_sample()
                negative_bank.append(anchor.clone())
                negative_bank.append(negative.clone())
            negative_bank = torch.stack(negative_bank).to(device)

            if make_memory_bank:
                return negative_bank.unsqueeze(1)

        if np.random.rand() < self.change_probability:
            self.change_tomogram()

        batch = []
        for _ in range(batch_size):
            anchor, positive, negative = self.get_sample()
            batch.append(torch.stack([anchor, positive, negative], dim=0))
        return torch.stack(batch, dim=1).unsqueeze(2).to(device)

    def change_tomogram(self):
        """
        Returns
        -------
        None

        This updates chooses a new tomogram, uploads its annotations and computes the connected components.
        Labels for different connected components are stored to speed up the process through avoiding computing
        the connected components multiple times.
        At least two connected components are needed in order to select the positive and negative samples.
        """
        self.num_features = 0
        while self.num_features < 2:
            self.chosen = np.random.choice(self.names).split('/')[-1]
            annotations = mrcfile.open(f'{self.path}/czi_membrain/{self.chosen}').data
            self.annotations = np.array(annotations, dtype=np.float32)
            tomo = mrcfile.open(f'{self.path}/czi/{self.chosen}').data
            self.tomo = np.array(tomo, dtype=np.float32)
            self.anchor_input_shape = determine_input_shape(self.pixel_size[self.chosen.split('.')[0]],
                                                            self.pixel_out, self.crop_size)
            if not os.path.exists(f'{self.path}/labeled_annotations/{self.chosen}'):
                self.labeled_membranes, self.num_features = scipy.ndimage.measurements.label(self.annotations)
                np.save(f'{self.path}/labeled_annotations/{self.chosen}', self.labeled_membranes)
            else:
                self.labeled_membranes = np.load(f'{self.path}/labeled_annotations/{self.chosen}')
                self.num_features = np.unique(self.labeled_membranes) - 1

    def get_sample(self):
        """
        Returns
        -------
        three torch.tensors.

        Anchor and positive are going to belong to the same connected component while negative is going to
        belong to a different one.
        Three samples will be taken from the same tomogram to avoid classifications being based on tomogram's
        particularities.
        """
        chosen_feature = np.random.choice(np.arange(1, self.num_features + 1))
        chosen_membrane = np.array(np.where(self.labeled_membranes == chosen_feature)).T
        anchor = self.get_crop(chosen_membrane, self.tomo, self.anchor_input_shape)
        positive = self.get_crop(chosen_membrane, self.tomo, self.anchor_input_shape)

        negative_feature = np.random.choice(
            np.setdiff1d(np.arange(1, self.num_features + 1), np.array([chosen_feature])))
        negative_membrane = np.array(np.where(self.labeled_membranes == negative_feature)).T
        negative = self.get_crop(negative_membrane, self.tomo, self.anchor_input_shape)

        anchor = torch.tensor(anchor)
        positive = torch.tensor(positive)
        negative = torch.tensor(negative)

        return anchor, positive, negative

    def get_crop(self, chosen_membrane: np.array, tomo: np.array, input_shape: tuple):
        """

        Parameters
        ----------
        chosen_membrane: positions describing the connected component from where to extract the crop.
        tomo: tomogram to extract the crop from.
        input_shape: size of the crop needed to achieve the desired output shape after pixel size normalization.

        Returns
        -------
        np.array with a subtomogram with the selected pixel size.
        """
        subtomo = np.random.choice(len(chosen_membrane))
        subtomo = chosen_membrane[subtomo]

        subtomo[0] = min(max(subtomo[0], input_shape[0] // 2), tomo.shape[0] - input_shape[0] // 2)
        subtomo[1] = min(max(subtomo[1], input_shape[1] // 2), tomo.shape[1] - input_shape[1] // 2)
        subtomo[2] = min(max(subtomo[2], input_shape[2] // 2), tomo.shape[2] - input_shape[2] // 2)

        subtomo = tomo[subtomo[0] - input_shape[0] // 2:subtomo[0] + input_shape[0] // 2,
                       subtomo[1] - input_shape[1] // 2:subtomo[1] + input_shape[1] // 2,
                       subtomo[2] - input_shape[2] // 2:subtomo[2] + input_shape[2] // 2]

        if (self.pixel_size[self.chosen.split('.')[0]] / self.pixel_out) < 1.0:
            subtomo = fourier_cropping(subtomo, self.crop_size, False)
        elif (self.pixel_size[self.chosen.split('.')[0]] / self.pixel_out) > 1.0:
            subtomo = fourier_extend(subtomo, self.crop_size, True)

        return subtomo
