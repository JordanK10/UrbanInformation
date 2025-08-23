from setuptools import setup, find_packages

setup(
    name="MSAInformationComparison",
    version="0.1.0",
    description="A Bayesian framework for inferring agent learning in a dynamic urban environment.",
    author="Jordan Kemp",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "arviz",
        "pymc",
        "matplotlib",
        "tqdm",
        "joblib",
        "census",
    ],
    python_requires=">=3.8",
) 