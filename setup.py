from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2",
               "rpy2>=3.5.5","torch>=1.9.0","SimpleITK>=2.2.0","nibabel>=5.0.0",
               "scipy>=1.7.1","tqdm>=4.64.1","scikit-learn>=1.2.1",]

setup(
    name="pkg_bnnstgp1",
    version="0.1.5",
    author="Ella he",
    author_email="ellahe@umich.edu",
    description="BNNSTGP",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Ketherine0/BNNSTGP",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)