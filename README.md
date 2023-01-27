# End2End-DLMRI

Here, we provide an implementation of the network described in our work

"Deep Supervised Dictionary Learning by Algorithm Unrolling - Application to Fast 2D Dynamic MR Image Reconstruction"

by A. Kofler, M.C. Pali, T. Schaeffter and C. Kolbitsch.

The method is a so-called physics-informed neural network which implements an algorithm which approaches the solution of the problem

![equation](https://latex.codecogs.com/svg.image?\underset{\mathbf{x},&space;\{\boldsymbol{\gamma}_j\}_j}{\mathrm{min}}&space;\frac{1}{2}\|\|&space;\mathbf{F}_I\mathbf{x}&space;&space;&space;-\mathbf{y}_I\|\|_2^2&space;&plus;&space;\frac{\lambda}{2}&space;\sum_{j=1}^{N_{\mathbf{d},\mathbf{s}}}&space;\|\|&space;\mathbf{R}_j^{\mathbf{d},&space;\mathbf{s}}&space;\mathbf{x}&space;-&space;\mathbf{\Psi}&space;\boldsymbol{\gamma}_j&space;\|\|_2^2&space;&plus;&space;\alpha&space;\sum_{j=1}^{N_{\mathbf{d},\mathbf{s}}}&space;\|\|&space;\boldsymbol{\gamma}_j\|\|_1)

by alternating minimization. The weights of the network - i.e. the atoms of the dictionary and the regularization parameters- can be trained in a supervised and physics-informed way.

## Code

A hopefully somewhat decent and clean version of the code will be released soon ;) Please forgive the time delay.

## Citing this work
For citing this work, please use

@article{kofler2022end2end_dlmri,
  title={Deep Supervised Dictionary Learning by Algorithmic Unrolling - Application to Fast 2D Dynamic {MR} Image Reconstruction},
  author={Kofler, Andreas and Pali, Marie-Christine and Schaeffter, Tobias and Kolbitsch, Christoph},
  journal={Medical Physics},
  volume={},
  number={},
  pages={In Press},
  year={2023},
  publisher={Wiley Online Library}
}
