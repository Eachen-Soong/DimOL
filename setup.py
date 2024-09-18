from setuptools import setup, find_packages

setup(
    name="neuralop1",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'wandb',
        'ruamel.yaml',
        'configmypy',
        'tensorly',
        'tensorly-torch',
        'torch-harmonics',
        'matplotlib',
        'opt-einsum',
        'h5py',
        'zarr',
        'einops',
        'torchvision',
        'tensorboard',
        'tqdm',
    ],
    # entry_points={
    #     'console_scripts': [
            
    #     ],
    # },
)
