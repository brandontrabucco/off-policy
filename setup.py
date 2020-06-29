from setuptools import find_packages
from setuptools import setup


setup(
    name='offpolicy',
    version='0.1',
    description='Off-Policy Reinforcement Learning Algorithms',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.2',
        'tensorflow-probability',
        'gym[mujoco]',
        'numpy',
        'dm-tree',
        'scikit-video'])
