---
layout: post
title: "Understanding Physics-Informed Learning from the Ground Up"
subtitle: "The Start of a Journey"
date: 2026-01-25
substack_url: "https://hannesvdc.substack.com/p/understanding-physics-informed-learning"
---

Physics-informed machine learning is a beautiful idea: learning a neural network that satisfies first-principles physics *automatically*. It replaces supervised learning with a fundamentally different approach. In supervised learning, we start from large datasets that contain known input-output relations $(x_i, d_i)$ and minimize either the averaged squared deviation

$$
    L(\theta) = \frac{1}{N} \sum_{i=1}^N \left(\text{NN}_{\theta}(x_i) - d_i\right)^2
$$

in case of a continuous output space, or the cross-entropy loss

$$
    L(\theta) = -\sum_{c=1}^C p_c(x_i, \theta) \log q_c
$$

for classification. Here, $\theta$ denotes the network weights and biases. In the latter case, $C$ is the number of classes, $c$ indexes a particular class, and $p_c$ are the network output probabilities for each class (typically produced by a softmax). The ground-truth probabilities $q_c$ are one-hot encoded for each input.

Physics-informed ML minimizes the mismatch with physics. The simplest formulation assumes we are working with a differential equation

$$
    \mathcal{L}[u, p](x, t) = f(x),
$$

in space and/or time. In the most general formulation, $\mathcal{L}$ is a (nonlinear) differential operator, $u(x,t)$ is the solution, $p$ is the parameter and $f$ is a forcing term. Boundary and initial conditions are included implicitly, yet no less important. For given parameters, boundary and initial conditions and forcing, a Physics-Informed Neural Network (PINN) represents the unique solution function $(x,t) \mapsto u(x,t)$ of the PDE. This function is learned by minimizing the squared residual of the PDE. Specifically, for batched inputs $(x_i, t_i)_{i=1}^N$, the physics-informed loss reads

$$
    L(\theta) = \frac{1}{N} \sum_{i=1}^N \left(\mathcal{L}\left(u_{\theta}(x_i,t_i)\right) - f(x_i) \right)^2,
$$

with $u_{\theta}$ the output of the neural network. 

To enforce the governing equations, the PINN output $u_{\theta}$ must be differentiable with respect to its inputs. This differentiability must be enforced at the design stage and, for example, rules out the non-differentiable ReLU activation function. In fact, $\tanh$ is the default activation function in physics-informed learning.

The advantage of PINNs is that, once trained, they represent the continuous solution that can be evaluated at any point $(x,t)$ in the space–time domain, without ever needing to define a mesh or grid. This is in sharp contrast to classical numerical solvers where the solution is only available at discrete nodes of the mesh.

Sadly, in its most basic form, a PINN only represents the solution corresponding to a single realization of the forcing terms, boundary conditions, and parameter values. Retraining is required when any of these elements change - making this whole endeavor very inefficient. More powerful are Physics-Informed Neural Operators (PINOs), which learn the solution operator itself, mapping parameters, boundary conditions, and forcing terms $(f, p)$ directly to the solution function $u$. PINOs are much more expressive but are larger and require a lot more attention during training. Yet, I believe it is worth investigating their performance in more detail.

Over the last six months or so, I’ve tried training Physics-Informed Neural Operators for a range of problems, without success. I feel like there is something I fundamentally don't understand about PINNS and I need to know what that is. To be fair, the cases have been far from easy: learning the full two-dimensional displacement field of a metallic plate under random boundary forcing, or inferring potential energy profiles of small molecules. Still, physics-informed learning should be able to handle these examples. 

This experience has made it clear to me that I need to go back to basics and really understand physics-informed learning at its core—starting with PINNs and PINOs themselves. I need to take one step back in order to take two forward. 

With this old saying in mind, I will start from a deliberately simple example and gradually increase the difficulty. My express goal is to use the knowledge, ideas and feelings I gather along the way to build a PINO that represents the solution operator of a second-order PDE in three spatial dimensions. Perhaps we can build neural models that are truly first-principles–based, without forcing any kind of PDE framework.

You’re invited to join me on this journey. Nothing is guaranteed except a lot of learning, plenty of frustration, and a rollercoaster of emotions. I promise only one thing: I’ll write it all down here.