import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='road',
     version='0.0.5',
     author="Peng Xiong and Zhi Chen",
     author_email="xiongpengnus@gmail.com",
     description="Robust Optimization with Array Data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/XiongPengNUS/road",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )
