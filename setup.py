from setuptools import find_packages
from setuptools import setup


setup(
    name='offpolicy',
    description='Off-Policy Reinforcement Learning Algorithms',
    license='MIT',
    version='0.1',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.2.1',
        'tensorflow-probability',
        'gym[mujoco]',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'click'],
    entry_points={
        'console_scripts': ('offpolicy=offpolicy.cli:cli',)})
