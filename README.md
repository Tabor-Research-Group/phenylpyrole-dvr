# phenylpyrole-dvr

The directories phpy_data and phpy_h2o_data contain jupyter notebooks that generate the plots and data used in the paper. 

The Fortran 90 program dvr_1d_periodic.f90 calculates the energies and wavefunctions for a periodic potential. The notebook indicates when the code should be executed. The user needs to provide their own matrix diagonalization subroutine (called house in the provided code). 

The program dvr_1d_periodic.f90 was compiled with GNU Fortran (GCC) 8.2.0.

The following versions of python packages were used in the data analysis notebooks.

python   3.6.9  

numpy  1.16.4

scipy   1.5.2 

matplotlib  3.1.1
