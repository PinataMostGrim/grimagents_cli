import setuptools
from pathlib import Path


readme_path = Path(__file__).with_name('readme.md')
with readme_path.open(mode='r') as f:
    long_description = f.read()

setuptools.setup(
    name="grimagents",
    version="2.1.1",
    description="Collection of command line applications that wrap Unity's Machine Learning Agents toolkit with more automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PinataMostGrim/grimagents_cli",
    author="PinataMostGrim",
    author_email="pinatamostgrim@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "pyyaml",
        "bayesian-optimization>=1.0.3,<=1.1.0"
    ],
    python_requires=">=3.6,<3.7",
    entry_points={
        'console_scripts': [
            'grimagents = grimagents.__main__:main',
            'grimwrapper = grimagents.training_wrapper:main',
            'grimsearch = grimagents.search:main'
        ],
    }
)
