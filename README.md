# FTPT
This is the code for vibrational finite-temperature perturbation theory.  
The related paper is still in preparation and the parallel versioin of finite-temperature perturbation theory for electrons is published:
J. Chem. Phys. 155, 094106 (2021) [DOI](https://doi.org/10.1063/5.0061384)

## Features
In ListofPTterm.py file, I designed the data structure to store algebraic expressions for second-order perturbation corrections to the grand potentials.
Each class stores the expressions with same energy differences. 

In FTPTeval.py file, I did the analytical derivation step by step :
* Start with zero-temperature MP2 energy correction: <N | V | M> / (E_N - E_M)
* Evaluate each MP2 energy by Born-Huang rules for all the combinations between force constants and energy difference. (for electrons, Slater-Condon rules).
* Store the terms with same energy difference into same ListofPTterm class.
* In each class, merge the terms with reversed-sign energy difference.
* Substitute quantum number I_m^n with Bose-Einstein distribution function f_m (for electrons, Fermi-Dirac distribution function).
* Drop the redundant terms, namely, the terms that if we switch the force constants, they turn into same expressions.
* Multiply each term with prefactors from vibrational Hamiltonian.
* Generate the full analytical expressions and run numerical test in numericaltest.py

## Incoming Feature
Generalize to:
* zero-temperature perturbation correction
* finite-temperature perturbaiton theory for electrons.
* higher-order force field
Adding second-quantization derivation for more concise expression.
