### Tactile Conditioned Video Prediction for Physical Robot Interaction

First Author: Willow Mandil
Second Author: Amir Ghalamzan E

## Requirements:
Use the venv at robotics/tactile_prediction/venv

## Dataset Description
Locate the dataset and its description at: 

To format the data apply:


## Model Architectures:
For the prediction models p and q we use the convolutional dynamic neural advection (CDNA) model described in finn2016unsupervised with the additional stochastic variational inference network described in babaeizadeh2017stochastic. Of course, the focus of this work is to show the effect on video prediction accuracy of introducing tactile data, so it in not essential to integrate with the best performing VP models at the time, however the model described appears to be one of the best in this class at time of writing nunes2020action. 


