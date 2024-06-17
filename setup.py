from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    install_requires = [
        line.strip() for line in f.readlines() if line.strip()
    ]

setup(
    name="bnnstgp",
    version="0.1",
    author="Ella he",
    author_email="ellahe@umich.edu",
    description="BNNSTGP",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Ketherine0/BNNSTGP",
    packages=find_packages(),
    install_requires=install_requires,  # Use install_requires instead of requirements
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
