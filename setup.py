from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='cosense3d',
    version=__version__,
    author='Yunshuang Yuan',
    author_email='-',
    url='-',
    license='MIT',
    # packages=find_packages(include=['config', 'dataset', 'dataset.toolkit', 'model', 'ops',
    #                                 'tools', 'utils']),
    packages=find_packages(include=['cosense3d', 'interface']),
    zip_safe=False,
)
