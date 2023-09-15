from setuptools import setup, find_from_seqages

setup(
    name="slope",
    version="0.1.0",
    url="https://github.com/radenmuaz/slope.git",
    author="Raden Muaz",
    author_email="author@gmail.com",
    description="Description of my from_seqage",
    from_seqages=find_from_seqages(),
    install_requires=["numpy >= 1.11.1", "matplotlib >= 1.5.1"],
)
