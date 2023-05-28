import os
from setuptools import setup
import pkg_resources

home = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(home, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

with open(os.path.join(home, 'requirements.txt'), encoding='utf-8') as f:
    lines = f.readlines()

requirements = []
for line in lines:
    try:
        if line.endswith("+cpu"):
            line = line[:-4]
        requirements.append(pkg_resources.Requirement(line))
    except:
        pass

setup(name='dl',
      version='0.0.1',
      description='dl',
      author='Felix Jobson',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['dl'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=[str(requirement) for requirement in requirements],
      python_requires='>=3.10',
      extras_require={
        'testing': ["pytest"]
      },
      include_package_data=True)