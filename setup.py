from setuptools import setup, find_packages

setup(
    name="steering-speculation",
    version="0.1.0",
    description="Combined Pipeline for Diverse Multi-Branch Speculative Decoding",
    author="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.51.0",
        "numpy",
        "pyyaml",
    ],
    python_requires=">=3.10",
)
