import math
import os
import sys
import numpy as np
import boto3
import torch
import webdataset as wds
import nibabel as nib
import pickle as pkl
import torchio as tio
import random
from tqdm import tqdm

start = 0
end = 100
patch_size = 13
num_frames = 8
sampling_rate = 1
duration = num_frames * sampling_rate
use_bspline = True
shuffle = True

# Connect to S3
s3 = boto3.client('s3')

# Set the bucket name and folder name
bucket_name = 'openneuro.org'

# List all folders in the parent directory
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')

# Extract the folder names from the response
folder_names = [x['Prefix'].split('/')[-2] for x in response.get('CommonPrefixes', [])]
folder_names = folder_names[start:end]
print(folder_names)
# random.shuffle(folder_names)

obj_key_list = []

for folder_name in tqdm(folder_names):
    # List all objects in the folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    # Process each object in the folder
    anat_subj = None
    for obj in response.get('Contents', []):
        obj_key = obj['Key']

        # if '_T1w.nii.gz' in obj_key or '_bold.nii.gz' in obj_key:
        #     print(folder_name, obj['Key'])
        if '_T1w.nii.gz' in obj_key: # Anatomical
            # Store subject number to verify anat/func match
            anat_subj = obj_key.split('/')[1]
            
            # Download the object to tmp location
            # filename = os.path.join('tmp', 'T1w.nii.gz')
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            # s3.download_file(bucket_name, obj_key, filename)

            # # store the head of anat_subj
            # anat_header = nib.load(filename).header
            
        elif '_bold.nii.gz' in obj_key: # Functional bold
            # Verify func/anat subject number match
            func_subj = obj_key.split('/')[1]
            if anat_subj != func_subj:
                # print('Incompatible subject number found.', anat_subj, func_subj)
                # raise ValueError('Incompatible subject number found.')
                pass
            else:
                obj_key_list.append(obj_key)

            # break
    

if shuffle:
    random.seed(0)
    random.shuffle(obj_key_list)

tar_folder = f"openneuro-{start}-{end}-ps{patch_size}-f{num_frames}-r{sampling_rate}{'-bspline' if use_bspline else ''}{'-shuffled' if shuffle else ''}"
tar_folder = os.path.join('/scratch', tar_folder)
os.makedirs(tar_folder, exist_ok=True)
sink = wds.ShardWriter(f"{tar_folder}/func-%06d.tar", maxcount=1000)

filename = os.path.join(tar_folder, 'tmp', 'bold.nii.gz')
os.makedirs(os.path.dirname(filename), exist_ok=True)

for obj_key in tqdm(obj_key_list):
    s3.download_file(bucket_name, obj_key, filename)


    t1 = tio.ScalarImage(filename)
    t1 = tio.ToCanonical()(t1)
    # t1.plot(output_path='tmp/bold.jpg')
    T, X, Y, Z = t1.shape
    lx, ly, lz = t1.spacing
    patch_length = []
    def valid_pl(l):
        sx = math.ceil(X * lx / l)
        sy = math.ceil(Y * ly / l)
        sz = math.ceil(Z * lz / l)
        npatches = sx * sy * sz
        if npatches <= 196:
            return l
        else:
            return 1e9

    for i in range(1, 20):
        patch_length.append(valid_pl(X * lx / i))
        patch_length.append(valid_pl(Y * ly / i))
        patch_length.append(valid_pl(Z * lz / i))
    # get the minimum patch size that will fit in 196 patches
    patch_length = min(patch_length) + 1e-6
    t1.set_data(t1.data.type(torch.float32))
    if use_bspline:
        resample = tio.Resample(patch_length / patch_size, image_interpolation='bspline')
    else:
        resample = tio.Resample(patch_length / patch_size)
    # print(t1)
    t1 = resample(t1)
    # print(t1)
    shape = t1.shape[1:]
    patch = (math.ceil(shape[0] / patch_size), math.ceil(shape[1] / patch_size), math.ceil(shape[2] / patch_size))
    # print(T, X, Y, Z)
    # print(lx, ly, lz)
    # print(patch_length)
    # print(patch)
    assert(patch[0] * patch[1] * patch[2] <= 196)
    final_shape = (patch[0] * patch_size, patch[1] * patch_size, patch[2] * patch_size)

    # print(shape)
    # print(final_shape)
    pad = tio.CropOrPad(final_shape)
    t1 = pad(t1)
    assert(t1.shape[1:] == final_shape)
    # t1.plot(output_path='tmp/bold_resampled.jpg')

    # filename = os.path.join('tmp', 'bold_reshaped.nii.gz')
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # t1.save(filename)

    # print(t1)
    # print(t1.data.shape)
    # print(t1.numpy().shape)
    data = t1.numpy().astype(np.float32)
    data /= np.mean(np.abs(data))
    data = data.astype(np.float16)
    T = data.shape[0]

    key_without_ext = obj_key.split('.')[0]
    for i in range(0, T - duration + sampling_rate, duration):
        data_slice = data[i:i+duration:sampling_rate, :, :, :]
        sink.write({
            "__key__": f'{key_without_ext}-{i}',
            "func.npy": data_slice,
        })

    # exit(0)


            
