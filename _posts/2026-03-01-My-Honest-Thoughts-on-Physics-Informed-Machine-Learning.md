---
layout: post
title: "The Deep Truth about Physics-Informed Learning"
date: 2026-03-01
substack_url: "https://hannesvdc.substack.com/p/the-deep-truth-about-physics-informed"
---

Over the past month I've been exploring physics-informed learning to understand how viable it has become as alternative to existing expensive solvers for PDEs. It has been a journey filled with learning, excitement, frustration and the occasional a-ha! moment. I really recommend taking a deep dive into any topic - and physics-informed ML in particular - because not only will you understand the subject much better, you will also start generating your own ideas and learn about related topics. It's really a story of self-development and building expertise!

Before continuing, let us define what physics-informed learning actually is. To me, physics-informed learning refers to any machine learning approach that solves a problem directly from first principles, rather than from observation or measurement data. The governing equations, be it in physics, chemistry, biology or any quantitative field, define the problem and a physics-informed model then solves it. The neural network is not trained to imitate data but tries to satisfy the underlying law as closely as possible.

In that sense, physics-informed ML is not limited to solving differential equations, although that was what it was originally proposed for [Raissi et. al.](https://www.sciencedirect.com/science/article/pii/S0021999118307125). In fact, I am very interested in exploring what physics-informed learning *means* for chemical reaction, biological systems, computational drug design, fundamental physics, and so on. Granted, many of these examples are ruled by differential equations but definitely not all. Could any of these fields be advanced by sprinkling some chemistry-informed learning, biology-informed learning, or generally pick-your-favorite-field-informed learning??

Physics-informed learning is a totally different paradigm from "data-informed" learning and should be viewed differently. It is almost ironic that, over the past month or so, I have only really applied it to differential equations, the field where PINNs were originally introduced.

## Back to the Roots
So let's talk about physics-informed learning in the context of differential equations. There are two broad approaches: physics-informed neural networks (PINNs) and physics-informed neural operators (PINOs). Starting from a differential equation in its most general form

$$
    \mathcal{L}\left( u, p, f(x)\right) = 0, \quad x \in \Omega
    \tag{1}
$$

where $\mathcal{L}$ is the differential operator, the objective is to predict the solution $u(x,t)$ *from the equation alone*. The differential operator encodes partial derivatives $\frac{\partial u}{\partial t}$, $\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$ and more. $p$ are the model parameters, and $f(x)$ represents any source / sink function or boundary condition. PINNS solve (1) for given values of the parameters $p$, known sources and boundaries, and a fixed initial condition $u_0(x)$. That is, a physics-informed neural network maps

$$
    (x, t) \longmapsto u_{\theta}(x, t).
$$

Nothing more, nothing less. Here $\theta$ are the weights and biases of the neural network. Because of their 'simplicity', PINNs are usually small networks that have been optimized for this task alone. Although the setup for PINNs might seem limited, it often happens in industry that one wants to solve the same PDE under identical initial and boundary conditions with just different parameters. PINNs are perfect for this task and are not as bloated as PINOs. On the other hand, physics-informed *operators* learn the full solution map

$$
    (x, t, p, f, u_0) \longmapsto u_{\theta}[p, f, u_0](x,t)
$$

and are *much* more general because they can evaluate the solution for any values of the inputs. That way, it is expected that PINOs learn much more of the physics but are harder to train. Generality comes at a cost. Also, PINOs typically use a more advanced architectures involving convolutional layers, pooling, LayerNorm and even Transformers (for advancing in time mapping)!

Contrary to most machine learning models, physics-informed learning does not use any data. Its objective is not to minimize some MSE or cross-entropy loss, rather it is to satisfy the differential equation (1) as much as possible by minimizing the expected residual

$$
    \mathbb{E}_x \, \mathbb{E}_t \, \mathbb{E}_p \, \mathbb{E}_{u_0} \, \mathbb{E}_f
\left[
\left\lVert
\mathcal{L}\!\left(u_{\theta}, p, f(x)\right)
\right\rVert^2
\right]
    \tag{2}
$$

Minimizing the residual between the network output and physics is a fundamentally different idea from learning behavior from data. It's benefit is that no training data needs to be available. The beauty of physics-informed learning is that we can *sample* $x$, $t$, $p$, $u_0$ and $f$ however we like / need to make the network learn. However, the downside is that evaluating the physics loss requires derivatives of the network output $u_{\theta}(x,t)$ to $x$ and $t$ (but not to $p$ or $u_0$ or $f$). Since most interesting physics uses second-order derivatives, evaluating these numerically (through automatic differentiation) can pose interesting issues with accuracy and memory requirements!

So what are the constituent architectural elements of PINNs and PINOs? If the input are just scalars like $x$, $t$ and $p$, a multi-layer perceptron (MLP) with just a few hidden layers is more than sufficient to approximate the solution $u(x,t)$. That is what early researchers used the most due to its simplicity. However, when the inputs are functions like $f$ and $u_0$, learning substantially improves with a stronger inductive bias. In practice, input functions are first discretized on a grid (or provided in that form) and then passed to the so-called `branch' network. Rather than using a plain MLP, this branch network typically consists of several convolutional layers, often combined with pooling or downsampling. Convolutions encode locality, translation equivariance, and hierarchical feature extraction; properties that naturally align with continuous functions and differential operations applied to them. A convolutional architecture therefore embeds structure about how physical systems tend to behave.

## We Don’t Do PINNs Because They Are Easy — But Because They Are Hard
Long story short, PINNs hold great potential. They were introduced to the literature in exactly this way: a completely new way for solving differential equations that is geometry-agnostic, mesh-free and works from first principles without the need for pre-existing observational data. These properties are exactly what got me so interested in this topic - but to this date, I don't believe physics-informed learning has satisfied that promise. Granted, every new research direction must start simple in order to get the basics right, but it is exactly this dichotomy that got me interested in exploring PINNs for myself. Personally, However, they still can with the right application. How? Here are my two cents.

Over the past four weeks I have applied physics-informed learning to increasingly complex differential equations:
1. Newton's law of heat transfer 
    
    $$
        \frac{dT(t)}{dt} + \kappa \left(T(t)-T_s\right) \quad T(0) = T_0
    $$

    for any time $t$, initial temperature $T_0$, rate constant $\kappa$ and heat bath temperature $T_s$;
2. The heat / diffusion equation
    
    $$
        \frac{\partial T(x,t)}{\partial t} = \kappa \frac{\partial^2 T(x,t)}{\partial x^2} \quad T(x,0) = T_0(x), \ T(x=0,1,t) = T_s
    $$

    for any time $t$, point $x$, rate constant $\kappa$, heat bath temperature $T_s$ (as boundary condition) but *fixed* initial temperature profile $T_0(x)$;

3. The full heat equation with general initial temperature profile $T_0(x)$ as input to the neural operator;

4. The 2D linear deformation (elastostatic) equation with a combination of Dirichlet and traction (Neumann) boundary conditions
    
    $$
        \nabla \cdot \sigma(u) = 0 \quad u(x=0,y) = 0, \quad \sigma(u(x=1,y)) \cdot n = (g_x, g_y)
    $$

    and free boundaries at $y=0$ and $y=1$. I will post soon about this example.

My experience is that it has been surprisingly hard to train a PINN on simple models like Newton's heat law and the diffusion equation, but it got significantly easier as the model's complexity increased. My experience may be one of survivor's bias; indeed, I invested a lot of time to get the earliest models to work, and the knowledge and experience I gained must have *scaled* to more complex models, but it has been an interesting journey nonetheless!

The main takeaway is that physics-informed learning is not automatically well-conditioned. Even simple equations like Newton's heat law can induce stiff optimization problems. There are essentially two reasons: (1) most of the interesting dynamics occurs near $t \approx 0$; (2) By design, the rate constant (and other parameters) $\kappa$ must span multiple orders of magnitude. For these two reasons, proper data sampling is an *essential* part of getting PINNs right, and so is including as much physics into the network *ansatz* to make learning easier. In fact, simple equations tend to suffer more from these drawbacks during training because there is less room to start making initial progress and less room to hide - all the details must be right at from the beginning for training to commence.

In fact, the guaranteed way to make physics-informed learning *fail* is to simply throw the physics residual at an optimizer and hope it magically learns the solution. One must understand the structure of the underlying problem and encode that structure directly into the architecture and output representation. Here are some general tips

- Make all parameters dimensionless. Instead of working with raw time $t$, use $\tau = \kappa t$. This simple transformation can dramatically reduce stiffness and improve conditioning.
    
- Normalize all inputs. This is closely related to the first point, but remains essential for stable optimization.
    
- Incorporate as much known physics as possible into the ansatz. If the temperature converges to a steady state $T_s$, bias the network so that reaching $T_s$ requires no effort. For example, one may set

    $$
        T(\tau) = T_s + \frac{\mathrm{NN}_{\theta}(\tau)}{1+\tau}
    $$

    for a generic decay, or use an exponential form when the physics is known to decay exponentially.
    
- Enforce Dirichlet and Neumann boundary conditions directly in the network output. For example, a Dirichlet condition $T(x=a,t) = T_s$ can be enforced via
    
    $$
        T(x,t) = T_s + (x-a)\,\mathrm{NN}_{\theta}(x,t).
    $$

    Similarly, a Neumann condition $\frac{\partial T(a,t)}{\partial x} = g$ can be enforced by
    
    $$
        T(x,t) = (x-a)g + (x-a)^2 \,\mathrm{NN}_{\theta}(x,t).
    $$
    
- Bias the sampling of $t$ (or $\tau$), $x$, parameters, and input functions toward regions where the nonlinear dynamics is strongest. These regions are the hardest to learn. An 80\% - 20\% biased sampling strategy often works surprisingly well.

 - Whenever possible, use the weak form of a second-order PDE. Second derivatives are expensive, require a lot of memory to store the taping graph and can be numerically less stable. Weak forms only require the first derivative ( at least in $x$ ) and have some nice interpretations to them. More about this later.
    
- Resample training data every epoch. Physics-informed networks *love* fresh samples. This effectively increases the training dataset at almost zero cost and reduces overfitting to particular collocation grids.
    
- Start with a first-order optimizer such as Adam to explore the landscape, and only switch to second-order methods (e.g. L-BFGS) once the solution is reasonably close. Jumping to a quasi-Newton method too early can amplify overfitting (a lot!).

## How Physics-Informed Learning can make a Difference.

It takes a lot to make physics-informed networks work, even on simple examples. Perhaps PINNs are not meant to replace classical solvers. Classical solvers have been optimized for over 100 years of research; they are already mature, efficient, and well-understood. Existing methods such as finite differences, finite volumes, and finite element methods are extremely good at solving many PDEs. They are stable, interpretable, and decades of numerical analysis have gone into understanding their conditioning and convergence. Replacing them with a neural network for the sake of novelty makes little sense. If we want physics-informed learning to be worth the effort, it must be applied to problems where classical approaches genuinely struggle.

In what form can PINNs be useful? It must be a setting where its biggest downsides never come up. The first downside is scale. The overhead of automatic differentiation, residual sampling, and complicated optimization strategies becomes justifiable for large, high-dimensional, or multiscale problems where traditional discretizations become prohibitively expensive. In small, well-behaved problems, classical methods will almost always win - and they should. Secondly, derivatives are expensive. Most strong-form PINNs require second-order derivatives of the network output. These are slow to compute and memory-intensive, especially in higher dimensions. This directly affects conditioning and stability. 

Fortunately, this is why weak formulations and energy principles are so attractive. 

What is a weak form? Given a (typically second order) PDE in its general formulation (also known as the *strong form*)

$$
    \mathcal{L}\left(u, p\right) = s(x)
    \tag{3}
$$

with Dirichlet conditions on one part of the boundary, $u(x) = f(x), x \in \partial \Omega_D$ and Neumann on the remaining boundary $\nabla u(x) \cdot n(x) = g(x), x \in \partial \Omega_N$. To derive the *weak form*, one multiplies the the strong form (3) with 'random' test functions $v$ and integrates over the whole domain $\Omega$

$$
    \int_{\Omega} \mathcal{L}\left(u, p\right) v dx = \int_{\Omega} s(x) v(x) dx + \int_{\partial \Omega_N} g(x) v(x) dx.
    \tag{4}
$$

If the original PDE holds, equation (4) must be true for any test function $v$ and vice versa. There are some technical restrictions on the class of test functions (they need to vanish on the Dirichlet boundary), but I won't go into that much further here. So how does the magic of the weak form come in?  One can reduce the order of derivatives required by *integration by parts*. The derivatives acting on $u$ are shifted onto $v$. This reduces equation (4) to 

$$
    \int_{\Omega} B\left(u, v\right) dx = \int_{\Omega} s(x) v(x) dx + \int_{\partial \Omega_N} g(x) v(x) dx.
$$

where $B$ is a *bilinear form* that acts on $u$ and $v$. Importantly, these bilinear forms only use information up the the *first derivative*. Instead of enforcing the PDE pointwise, weak formulations enforce it in an averaged or variational sense. Conditioning improves tremendously when switching from hard and steep point-wise losses to averaged and global losses, making training significantly more stable.

Instead of using the strong residual (2), one could minimize the weak residual

$$
    \mathbb{E}_x \ \mathbb{E}_t \ \mathbb{E}_p \ \mathbb{E}_{u_0}  \mathbb{E}_f\left[\sum_{n=1}^N\left(\int_{\Omega} B\left(u, v_n\right) dx - \int_{\Omega} s(x) v_n(x) dx - \int_{\partial \Omega_N} g(x) v_n(x) dx\right)^2\right].
$$

over a series of test functions $\{v_n\}_{n=1}^N$. Minimizing the weak and strong forms give the same solution but the former is much better conditioned. 

At this point, however, something interesting happens. In classical finite-element methods, the test functions $v_n$ are local basis functions tied to a mesh: piecewise polynomials ('hat functions') supported on small elements. The integrals are computed using numerical quadrature over those elements. Once we choose such local basis functions, introduce elements, and assemble contributions, we have essentially reconstructed the finite-element method.

There is a certain irony here. To avoid computing second derivatives via automatic differentiation, we move to a weak formulation. But implementing a classical weak formulation typically requires precisely what PINNs were promised to avoid: meshes, element connectivity, and quadrature rules. Furthermore, meshes are not always easy. Generating high-quality meshes for complex geometries is a nontrivial task. In fluid dynamics, for instance, classical computational fluid dynamics (CFD) simulations require extremely fine meshes in regions of turbulence or high Reynolds number. Boundary layers, vortical structures, and multiscale dynamics demand local refinement. The mesh becomes dense exactly where the physics becomes hard.

If weak formulations are desirable but local mesh-based test functions are not, what are we left with? An alternative is needed. Here is the main idea: weak forms do not inherently require local basis functions, they merely require test functions. Instead of compactly supported ``hat'' functions on triangles, one could use global basis functions: Fourier modes, orthogonal polynomials, wavelets, or other multiscale structures. There is a whole zoo of global functions!

In fact, one might exploit the strengths of traditional computational methods and PINOs on their own domain: using global basis functions and PINOs in regions with strong dynamics, and keep using the finite element method with a coarser mesh in the rest of the domain.  In such a setting, high resolution is no longer achieved through spatial mesh refinement, but through a set of global test functions of different 'frequencies'. High-frequency test functions are necessary for a high resolution of turbulence and other strong dynamic effects; the small scales do not disappear. In fact, this is what a growing class of spectral methods tries to achieve.

## Closing Thoughts
After a month of experimenting with physics-informed learning, my conclusion is neither blind optimism nor dismissal. PINNs are not magic solvers that will replace a century of numerical analysis overnight. Nor are they useless toys. They occupy a much more interesting middle ground. Physics-informed learning is, at its core, numerical analysis disguised as an optimization problem. Conditioning issues, stiffness, scaling are classical numerical issues that reappeared, they are not machine learning problems only.

Strong-form PINNs expose this reality brutally: they can be extraordinarily expensive and surprisingly unstable, especially when second derivatives enter through automatic differentiation. Weak formulations, in contrast, often lead to much softer and better conditioned learning problems. 

This is also where I think physics-informed learning can truly shine. Not by replacing finite element or CFD solvers that have been refined for over a century, but by *augmenting* them. In particular, I believe in hybrid strategies through domain decomposition: use finite elements, finite volumes, or other grid-based methods in regions where a coarse discretization is sufficient, but couple them to weak-form PINOs with global test functions (or spectral bases) in regions that would otherwise demand an extremely fine mesh.

I am sure there are many other regimes where PINNs and PINOs can provide real computational value. This hybrid, variational perspective is simply where I currently see the most immediate path to impact.

This leaves me with just one question

<p align="center"><em>Quo vadis, physics-informed learning?</em></p>

I’m curious what you think. Where do you believe physics-informed learning will actually move the needle in computational science?