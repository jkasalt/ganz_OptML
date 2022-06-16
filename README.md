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
└── notebook_xy.ipynb



README.md
```

TODO


## Authors

* [Luca Bracone](https://github.com/jkasalt)
* [François Charroin](https://github.com/FrancoisCharroin)
* [Thomas Rimbot](https://github.com/Thomas-debug-creator)