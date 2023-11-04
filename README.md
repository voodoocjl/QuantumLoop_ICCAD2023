# ICCAD

This repository contains the code for 2023 Quantum Computing for Drug Discovery Challenge at ICCAD.

```
requirements:

qiskit == 0.44.1
qiskit-aer == 0.12.2
mitiq == 0.29.0
numpy == 1.23.5
pandas == 2.0.3
```
simulator.py is recommend to run for the test of a batch of random seeds.

To run the code with argument ```--noise``` specifying the noise model ```1 (or 2, 3)``` and `--seeds` specifying random seeds taken. 
For example, if you want to use seeds 20, 21,23 to test the noise model 3, type

```
python simulator.py --noise 3 --seeds 20 21 23

```
Default noise model is 1 and default seeds is 20, 21, 30, 33, 36, 42, 43, 55, 67, 170

Output will be saved in correpoding csv file, such as result_1.csv for noise model 1 and so on.

simulator_notebook is designed to detail every steps of the algorithm.
