from setuptools import setup, find_packages

setup(
    name="slope",
    version="0.1.0",
    url="https://github.com/radenmuaz/slope.git",
    author="Raden Muaz",
    author_email="author@gmail.com",
    description="An automatic differentiation middleware",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.11.1",
        "matplotlib >= 1.5.1",
        "onnx >= 1.14.1",
        "onnxruntime >= 1.16.0"],
)
