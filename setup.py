import setuptools
from setuptools import setup

setup(
    name='gfit',
    version='0.2',
    url='https://github.com/samthiele/gfit',
    license='MIT',
    author='Sam Thiele',
    author_email='sam.thiele01@gmail.com',
    description='Fast multi-gaussian curve fitting',
    long_description='Convert spectra or other signals to the sum of multiple symmetric or asymmetric gaussian functions.',
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering"
        ],
    keywords='gaussian data fitting spectral features',
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'numba', 'tqdm'],
    project_urls={  # Optional
            'Source': 'https://github.com/samthiele/gfit',
        },
)
