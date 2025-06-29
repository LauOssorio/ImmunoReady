from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='ImmunoReady',
      version="0.0.1",
      description="Assessing if a peptide is a good substance for cancer vaccine",
      license="MIT",
      author="LO, BG, AL, JBF",
      author_email="ossoriolaura@gmail.com",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
