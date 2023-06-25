import nibabel as nib
import os
import numpy as np

import datasets

import boto3


_DESCRIPTION = """\
fMIR dataset from openneuro.org
"""

class fMRIConfig(datasets.BuilderConfig):
    """Builder Config for fMRI"""
 
    def __init__(self, data_url, num_datasets=[10, 1, 1], num_frames=8, sampling_rate=1, **kwargs):
        """BuilderConfig for fMRI.
        Args:
          data_url: `string`, url to download the zip file from.
          **kwargs: keyword arguments forwarded to super.
        """
        super(fMRIConfig, self).__init__(**kwargs)
        self.data_url = data_url
        self.num_datasets = num_datasets
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate

class fMRITest(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        fMRIConfig(name="test1", data_url="openneuro.org", version=VERSION, description="fMRI test dataset 1", ),
    ]

    DEFAULT_CONFIG_NAME = "test1"  # It's not mandatory to have a default configuration. Just use one if it make sense.


    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "test1":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            # features = datasets.Features(
            #     {
            #         # "func": np.ndarray(shape=(65,77,65,self.config.duration)),
            #         "func": datasets.Array4D(shape=(None,None,None,self.config.duration), dtype='float32'),
            #     }
            # )
            features = None
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
        )

    def _split_generators(self, dl_manager):

        # Connect to S3
        s3 = boto3.client('s3')

        bucket_name = self.config.data_url
        
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')

        folder_names = [x['Prefix'].split('/')[-2] for x in response.get('CommonPrefixes', [])]
        print(len(folder_names))

        ndatasets = self.config.num_datasets
        if isinstance(ndatasets, int):
            ndatasets = [ndatasets, 10, 10]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "bucket_name": bucket_name,
                    "folder_names": folder_names[:ndatasets[0]],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "bucket_name": bucket_name,
                    "folder_names": folder_names[ndatasets[0]:ndatasets[0] + ndatasets[1]],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "bucket_name": bucket_name,
                    "folder_names": folder_names[ndatasets[0] + ndatasets[1]:ndatasets[0] + ndatasets[1] + ndatasets[2]],
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, bucket_name, folder_names):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        s3 = boto3.client('s3')
        tmp_dir = os.path.join('tmp', folder_names[0]) if len(folder_names) > 0 else 'tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        anat_file = os.path.join(tmp_dir, 'T1w.nii.gz')
        func_file = os.path.join(tmp_dir, 'bold.nii.gz')
        
        duration = self.config.num_frames * self.config.sampling_rate

        for folder_name in folder_names:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
            for obj in response.get('Contents', []):
                obj_key = obj['Key']
                if '_T1w.nii.gz' in obj_key: # Anatomical
                    # Store subject number to verify anat/func match
                    anat_subj = obj_key.split('/')[1]
                    
                    # Download the object to tmp location
                    s3.download_file(bucket_name, obj_key, anat_file)

                    # store the head of anat_subj
                    anat_header = nib.load(anat_file).header
                elif '_bold.nii.gz' in obj_key: # Functional
                    func_subj = obj_key.split('/')[1]
                    if func_subj == anat_subj:
                        s3.download_file(bucket_name, obj_key, func_file)
                        func = nib.load(func_file).get_fdata().astype('float16')
                        func = np.transpose(func, (3, 0, 1, 2)) # T, X, Y, Z
                        shape = func.shape
                        # print(f"{obj_key}", shape)
                        for i in range(0, shape[0] - duration + self.config.sampling_rate, duration):
                            func_slice = func[i:i+duration:self.config.sampling_rate, :, :, :]
                            # print(f"{obj_key}-{i}", func_slice.shape)
                            yield f"{obj_key}-{i}", {
                                "func": func_slice,
                            }
            


