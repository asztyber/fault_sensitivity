# fault_sensitivity

This repository contains the code accompanying the paper:
A. Sztyber-Betley On models sensitivity to faults - theoretical versus
practical results in data-driven models submitted to The 35th International Conference on Principles of Diagnosis and Resilient Systems (DX'24)

## File structure
Notebooks to explore:
The code used to generate the results presented in the paper is available in three notebooks:
- simulation_simple_ex1.ipynb - system simulation and analysis of residuals generated with linear autoregressive models based on Model Structures and residuals generators based on Minimally Structurally Overdetermined Sets, Example 1
- simulation_simple_ex2.ipynb - system simulation and analysis of residuals generated with linear autoregressive models based on Model Structures and residuals generators based on Minimally Structurally Overdetermined Sets, Example 2
- rnn_residuals.ipynb - residuals based on recurrent state space models implemented with PyTorch for Example 1
- parameter_sensitivity.ipynb - analysis of the sensitivity of the residual generator to change in the model parameters for Example 1

Additional files:
- ResGen*.py - code for residual generators
- models.py - recurrent state space models implemented with PyTorch