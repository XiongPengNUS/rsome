import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='rsome',
     version='0.0.3',
     author="Peng Xiong, Zhi Chen, and Melvyn Sim",
     author_email="xiongpengnus@gmail.com",
     description="Robust Stochastic Optimization Made Easy",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/XiongPengNUS/rsome",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )
