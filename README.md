# Integrated Machine Learning and Finite Element Analysis for Rail-Wheel Wear Evaluation

## Overview
[cite_start]This repository contains the dataset and MATLAB source code to reproduce the findings presented in the paper *"Integrated Machine Learning and Finite Element Analysis for Rail-Wheel Wear Evaluation Based on Stress and Contact Mechanics"*[cite: 2783]. 

[cite_start]The project proposes a hybrid framework that integrates Finite Element Analysis (FEA) with multiple machine learning regression models to predict wear-related parameters in rail-wheel interactions[cite: 2829]. [cite_start]Using a dataset of 1,500 samples generated from a 3D parametric FE model [cite: 2830, 2835][cite_start], we train and evaluate six distinct algorithms to predict contact pressure, maximum stress, and contact area[cite: 2816].

## Repository Contents
* `x.mat`: The input dataset containing 1,500 samples of geometric profile variations. The 5 features are: `Rail_A (mm)`, `Rail_B (mm)`, `Wheel_A (mm)`, `Wheel_B (mm)`, and `Wheel_C (mm)`.
* [cite_start]`OutPut.mat`: The corresponding mechanical outputs extracted from the FEA simulations[cite: 2829]. The 3 target variables are: `Contact Area (mm^2)`, `Max. Pressure (MPa)`, and `Max. Stress (MPa)`.
* `MAIN_regression_R1.m`: The primary MATLAB script. It performs the following tasks:
    * Loads and preprocesses the dataset (70% training, 30% testing split).
    * Performs Bayesian Optimization to automatically tune the hyperparameters.
    * [cite_start]Trains six regression models: Gaussian Process Regression (GPR), Support Vector Machines (SVM), Decision Trees (DT), Linear Regression (LR), Non-linear Regression (NLR), and Artificial Neural Networks (ANN)[cite: 2817].
    * Evaluates the models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination (R²).
    * Automatically generates and exports high-resolution scatter plots (Actual vs. Predicted) for all algorithms.

## Prerequisites
To run the code, you will need **MATLAB** installed on your machine along with the following official toolboxes:
* **Statistics and Machine Learning Toolbox** (Required for GPR, SVM, DT, LR, and NLR models).
* **Deep Learning Toolbox** (Required for the `feedforwardnet` ANN model).

## How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YourUsername/Rail-Wheel-Wear-ML-FEA.git](https://github.com/YourUsername/Rail-Wheel-Wear-ML-FEA.git)
