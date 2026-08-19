---
layout: post
title: "Agentic Computational Science"
date: 2026-07-23
substack_url: "https://hannesvdc.substack.com/p/agentic-computational-science"
---

# Agentic Computational Science

Agentic Artificial Intelligence took the world by storm at the end of 2025, and we are still coming to grips with it. Agentic AI changed the software engineering workflow almost overnight. Before agents, even the most efficient AI users worked in a clunky loop: type a prompt into ChatGPT, wait for an answer, copy the generated code into an editor, run it manually, paste back any error messages, and repeat. This workflow increased my productivity by at least a factor of two, but it also became tedious very quickly.

Agentic AI changed that. Tools like Claude Code can edit files directly, run tests, inspect errors, modify the implementation, and iterate without constant copy-pasting and waiting periods where you can do nothing else. The human is still in the loop, but the loop has fundamentally changed. Instead of manually channeling information between a chat interface and the code editor, you can describe what you want, watch the agent work, and occasionally course-correct when it goes off in an unexpected direction. Most importantly, it is possible to do other things while waiting.

Agentic AI is not limited to writing code. I recently wrote a Prospectus-Agent that scours the internet for potential clients for a business, looks up email addresses, guesses some and validates, writes the outreach email and then sends it using the Gmail API. This agent reduces the search for potential clients from hours a day to just pressing a button. (Here is the [link](https://github.com/hannesvdc/Prospectus-Agent) to the open source project for those interested).

This is a real step change. If ‘old’ (I know...) AI gave someone a two- or three-fold productivity boost, agentic AI feels like another large multiplier on top of that. Not because the underlying models became that much better, but because of the coding and tooling harness around the LLM. Agentic AI closes the development loop much more tightly than any interface-based tool ever could.

## Agents *x* Computational Science?
Computational science is full of slow, repetitive, semi-manual loops. We set up simulations, launch jobs, wait for results, inspect logs, generate plots, diagnose errors, tune parameters, and launch the next run. Even when simulations are slow and expensive, they are often not the bottleneck.  The human in the loop is, and the reasons are obvious: we simply have other things to do like attending meetings, spending time with our families, sleeping and working on other projects. We are not always ‘on’ in the same way a well-managed AI agent can be.

In this post I want to explore the question

> What would it mean to bring agentic AI into computational science?

And share some of my experiences with recent work.

Computational science is different in one major way. Unlike mainstream software engineering, writing the code is not the bottleneck. Libraries for high-performance numerical solvers, molecular dynamics simulation, finite element methods and many more have already been written. Some of these libraries are more than four decades old and rely on highly optimized C, Fortran and CUDA code. It would be very surprising to top them in terms of speed and efficiency, but nothing is impossible of course.

Agents’ most valuable impact wouldn’t be in the coding layer, but in the research and application layers. Think about a small collection of agents monitoring simulations, diagnosing failures, analyzing intermediate results, adjusting hyperparameters, and keeping the research loop going. This seems like the real promise of agentic computational science and still leaves a large role for humans.

## CFD as an Example
The most direct impact may be in computational fluid dynamics (CFD). Large-scale simulations of the Navier–Stokes equations, especially in turbulent regimes, can take days or even weeks to complete and consume substantial computational resources in the process. Yet a successful outcome is not guaranteed. Mesh size, adaptive refinement, solver tolerances, time-step bounds, physical parameter choices, preconditioners, and boundary conditions can all have a nonlinear effect on whether a simulation converges, how long it takes, and how much compute it consumes.

To give a simple example, reducing the mesh size by a factor of two in a three-dimensional simulation does not merely double the cost. In three dimensions, halving the mesh spacing increases the number of grid cells or mesh elements by roughly a factor of eight. That already means more unknowns, more memory, and larger linear or nonlinear systems to solve.

But this is only the beginning. The larger system may be harder to solve, even if the matrix is sparse. The conditioning can change, the preconditioner may become less effective, Krylov solver settings that worked well on the coarser mesh may no longer transfer, and the smaller mesh size can enforce much stricter stability constraints. In practice, a seemingly simple mesh refinement can easily trigger a cascade of new numerical issues.

These choices are exactly the kind of thing CFD researchers deal with every day: change one simulation knob, adapt related (hyper)parameters and assess impact on the speed of convergence and accuracy. Fine-tuning the setup can take days, if not weeks. And it has to be redone for every new problem.

This is where agentic AI can become extremely powerful if used well. A failed simulation is not a total failure. It provides useful information about how to choose the next set of parameters for the next simulation. It is not uncommon to spend hours going through simulation logs to figure out what went wrong, and then to update the simulation with improved parameters. 

Agents can totally do this, and much faster! An AI agent can start a simulation, wait for it to converge or fail, read the logs and write testing scripts. These testing scripts can be standalone to be rerun by the user at a later point, or they can be tiny scripts that never even make it to the hard drive and are executed instead directly in the terminal. Anyone who has used Claude or Codex will have seen the agent run inline scripts, read the results and make decisions on output the user has never seen.

However, there is an underappreciated aspect of cost. Unlike traditional software engineering where executing code is ‘cheap’, running CFD experiments is typically very expensive. One CFD run can easily commit to thousands of GPU hours. An effective computational science agent cannot simply be rewarded for finding the most accurate answer. It is paramount the agent has a clear scientific objective, a notion of numerical accuracy, computational cost, and uncertainty. I am very interested to see how this problem will be solved in practice.

## What is Left for Humans?
Research is not just running many experiments and seeing what works. AI agents can act on specific instructions like “Find the optimal hyperparameter combination” or “Which discretization scheme works best here?”, but it does not know *what* to research, it does not know what is important in the real world. Giving research a direction is still a fundamentally human job; one that likely won’t ever go away. If you define your job as running experiments, then sure, AI will take it, but that was never your real role. Think of agentic AI as a very good junior developer or junior researcher. They can do a ton of work on their own, but they don’t (yet) know why it is important, or how the work fits within the broader vision of a research group or company. I don’t believe (agentic) AI will take this away. 

This is where I think the human role will remain strongest. We decide what questions are worth asking, because the world is human. Science will become increasingly automated – as it has been over the past decades – but the reasons for doing science will never go away. 
