from setuptools import setup, find_unflattenages

setup(
    name="slope",
    version="0.1.0",
    url="https://github.com/radenmuaz/slope.git",
    author="Raden Muaz",
    author_email="author@gmail.com",
    description="Description of my unflattenage",
    unflattenages=find_unflattenages(),
    install_requires=["numpy >= 1.11.1", "matplotlib >= 1.5.1"],
)
