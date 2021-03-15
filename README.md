# Auriga simulations
This repository can be used to compare the predictions of the Auriga simulations with the observations. For more details, see the paper by Verma et al. (2021).

**Prerequisites**
1. Usual python libraries
2. astropy [https://www.astropy.org/]
3. galpy [https://docs.galpy.org/en/v1.6.0/]
4. skycats
5. seaborn [https://seaborn.pydata.org/]

**Instructions**
If you have access to AST machines, copy the folder */iSIMBA/archive/raw/kuldeep/data* to your work directory for a quick test. The folder contains observed data as well as certain metadata to run *auriga.py* in just few minutes. Running the code for the first time (without metadata) with a mock catalogue can take about a day on a modern desktop with 8 threads. 
