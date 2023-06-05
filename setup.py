from setuptools import setup, find_packages

setup(
    name='slop',
    version='0.1.0',
    url='https://github.com/radenmuaz/slope.git',
    author='Raden Muaz',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)