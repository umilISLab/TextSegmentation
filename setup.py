from setuptools import setup


# Function to read dependencies from requirements.txt
def get_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


# Use the get_requirements function to get the dependencies
install_requires = get_requirements()

setup(
    name="textsegmentation",
    url="https://github.com/umilISLab/TextSegmentation",
    author="ISLab",
    author_email="islab@islab.di.unimi.it",
    packages=["textsegmentation"],
    install_requires=install_requires,
    version="0.1",
    license="MIT",
    description="--TODO--",
    long_description=open("README.md").read(),
)
