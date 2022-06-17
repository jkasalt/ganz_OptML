# Improving minimax optimizers with modified differential equations

This project is a study of the Discrete Time Algorithms emerging from $O(s^r)$-resolutions of Ordinary Differential Equations, as done in
[http://arxiv.org/abs/2001.08826], namely: 
- Gradient Descent Ascent (GDA)
- Optimistic Gradient Descent Ascent (OGDA)
- Extra Gradient Method (EGM)
- Jacobian Method (JM)

These algorithms are tested on a variety of loss functions $L(x,y)$
applied to a min-max problem of the form $min_x max_y L(x,y)$. The losses are of 3 general types:
- The basic loss $L(x,y) = xy$
- Bilinear losses of the form $L(x,y) = x^TAy$, where $A, x, y$ can be multidimensional
- Convex-concave losses


## Repository Structure
This repository presents the following structure:
```
Documentation
├── Report.pdf
├── lecture12a_gans_annotated.pdf
└── paper.pdf

Images

Implementation GANs
├── main.py
├── models.py
├── optimizers.py
└── ppm.py

Notebooks
├── notebook_convex_concave.ipynb
├── notebook_xAy.ipynb
├── notebook_xy.ipynb
└── odes.ipynb

README.md
```

The `Documentation` folder contains the report for this project, the main reference paper, and a copy of the lecture on GANs from the Machine Learning course at EPFL, since GANs are an application of min-max problems.

The `Images` folder contains the pictures used in the report. These pictures can be obtained from the scripts.

The  `Implementation GANs` folder contains scripts that we wrote to apply the optimizers to GANs. In particular, there is an implementation of PPM and EGM. In the end, these scripts were not used for the main study.

The `Notebooks` folder contains the main notebooks used in this study. The scripts used to investigate convex-concave losses, bilinear losses, and the special case of $L(x,y) = xy$. These scripts are rather simple and can be run without any special configuration need.


## Authors

* [Luca Bracone](https://github.com/jkasalt)
* [François Charroin](https://github.com/FrancoisCharroin)
* [Thomas Rimbot](https://github.com/Thomas-debug-creator)