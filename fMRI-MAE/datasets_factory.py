import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator

import torchio as tio
import datasets


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        # TODO: add augmentation for fMRI
        self.transform = transforms.Compose([
            tio.CropOrPad((65, 78, 65)),
            # tio.RandomFlip(axes=('LR',)),
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=False, default_pad_value='otsu'),
            # tio.RandomAffine(scales=(0.9, 1.1)),
            # tio.RandomNoise(std=(0, 0.1)),
            # tio.RandomBlur(std=(0, 0.1)),
            # tio.RandomBiasField(coefficients=(0, 0.1)),
            ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        # print(images['func'].shape)
        process_data = self.transform(images['func'])
        # print(process_data.shape)
        return {'func': process_data, 'mask': self.masked_position_generator()}

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = datasets.load_dataset('fMRI-openneuro', 'test1', 
                                    num_datasets=args.data_ndatasets, 
                                    num_frames=args.num_frames,
                                    sampling_rate=args.sampling_rate,
                                    streaming=True)
    dataset = dataset.map(transform)
    print("Data Aug = %s" % str(transform))
    return dataset['train']
