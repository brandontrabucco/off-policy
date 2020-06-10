from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.2',
                     'tensorflow-probability',
                     'gym[mujoco]',
                     'numpy',
                     'dm-tree']


PACKAGES = [package
            for package in find_packages() if
            package.startswith('sac')]


setup(name='sac',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=PACKAGES,
      description='Soft Actor Critic')
