from setuptools import setup

setup(
    name='gfit',
    version='0.1',
    packages=['gfit'],
    url='',
    license='',
    author='Sam Thiele',
    author_email='',
    description='Fast multi-gaussian curve fitting',
    install_requires = ['numpy','scipy','numba','tqdm'],
)
