# Improving minimax optimizers with modified differential equations

## Contributors: Luca Bracone, Thomas Rimbot, François Charroin

This project is an implementation of the optimizers found on
[http://arxiv.org/abs/2001.08826] using a GAN architechture trained on a
variety of datasets.

# Steps
- Implement GDA, OGDA and EGM to work on R2 with the L(x,y) = xy
- Look for Hessian-Gradient product methods and implement JM for this loss
- Compare these algos: convergence, number of steps -> visualization in 2d plane
- Generalize to bilinear loss L(x,y) = xAy, and study convergence properties as a function of the matrix
- Generalize to more exotic convex/concave losses
- Apply to GANs with MNIST