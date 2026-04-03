import setuptools
import os


# Function to read the version from version.py
def get_version():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_dir, 'src/torch_cnmf/version.py')) as f:
        locals = {}
        exec(f.read(), locals)
        return locals['__version__']

setuptools.setup(
    name="torch-cnmf",
    version=get_version(),
    author="Alexandra Mo",
    author_email="ymo@stanford.edu",
    description="Torch-based cNMF adapted from Dylan Kotliar's cNMF",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EngreitzLab/cNMF_benchmarking/tree/main/cNMF_benchmarking_pipeline/Inference/torch-cNMF",
    project_urls={
        "Bug Tracker": "https://github.com/EngreitzLab/cNMF_benchmarking/tree/main/cNMF_benchmarking_pipeline/Inference/torch-cNMF",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'cnmf = torch_cnmf:main',
        ],
    },
    install_requires=[
        'scikit-learn>=1.0',
        'anndata>=0.9',
        'scanpy',
        'pandas',
        'numpy',
        'matplotlib',
        'palettable',
        'scipy',
        'pyyaml',
        'torch',
        'muon',
        'nmf-torch',
    ]
)
