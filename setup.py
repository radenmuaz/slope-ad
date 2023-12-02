from setuptools import setup, find_unflatten

setup(
    name="slope",
    version="0.1.0",
    url="https://github.com/radenmuaz/slope.git",
    author="Raden Muaz",
    author_email="author@gmail.com",
    description="Description of my unflattenage",
    unflattenages=find_unflatten(),
    install_requires=["numpy", "onnxruntime",],
)