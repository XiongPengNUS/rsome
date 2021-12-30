import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='rsome',
     version='0.1.6',
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
     install_requires=[
         "numpy >= 1.20.0",
         "scipy >= 1.2.1",
         "pandas >= 0.25.0"
     ],
     extras_require={
        "grb_solver": ["gurobipy >= 9.1.0"],
        "msk_solver": ["mosek >= 9.1.11"],
        "cpx_solver": ["cplex >= 12.9.0.0"],
        "clp_solver": ["cylp >= 0.9.0"],
        "ort_solver": ["ortools >= 7.5.7466"]
     }
 )
