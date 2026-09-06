---
layout: post
title: "AI Agents as Numerical Analysts"
date: 2026-08-18
substack_url: "https://hannesvdc.substack.com/p/ai-agents-as-numerical-analysts"
---

Yesterday was the most fun I have had in a long time. After spending weeks reading about AI agents, MCP servers, structured outputs, context engineering, and writing about [Agentic Computational Science](https://www.hvandecasteele.com/blog/agentic-computational-science/), I finally built one. Not another chatbot or a simple LLM wrapper, but an AI agent that acts as a numerical analyst and solves partial differential equations.

Let's dive right into it.

The setup is simple. The goal for the agent is to solve the one-dimensional heat equation

$$
\frac{\partial T}{\partial t}(x,t)
=
\alpha
\frac{\partial^2 T}{\partial x^2}(x,t)
+
f(x,t),
\tag{1}
$$

on the domain $x \in [x_{\min}, x_{\max}]$ over the time window $t \in [0,1]$ seconds.

From a physical point of view, $\alpha$ is the diffusion constant that determines how quickly heat dissipates from one location to the next, while $f(x,t)$ is a forcing term representing an external heat source.

The agent has access to only two solver parameters:

- the number of spatial grid points, $n_x$,
- the number of time steps, $n_t$.

Using only these two knobs—and a solver whose implementation is completely hidden from the language model—the agent must discover the optimal combination $(n_x, n_t)$ that reduces the numerical error below a user-specified tolerance.

The agent knows nothing more than the fact that it is solving a heat equation and the name of the benchmark problem. It must reason about discretization errors, timestep stability, and the influence of the forcing term entirely from the outcomes of its own experiments.

Our numerical solver uses central finite differences for the spatial derivative and the forward (explicit) Euler method for time integration. Importantly, the AI agent does **not** know this. As far as the language model is concerned, the underlying implementation could just as well be finite differences, finite volumes, or finite elements.

## The Agentic Loop

The agent must iteratively suggest new discretization parameters $(n_x, n_t)$, call a tool to solve the PDE, call another tool to compute the error, analyze the error, and suggest an updated parameter set. Importantly, it has a limited budget of **10 iterations**.

Here is the full initial system prompt:

> You are a numerical analyst. Your goal is to solve the heat equation of the form
>
> $$
> u_t = \alpha u_{xx} + f(x,t)
> $$
>
> subject to (potentially nonzero) Dirichlet boundary conditions. The forcing term $f(x,t)$ is also not known to you, but you could find a hint in the model definition.
>
> Your goal is to solve the PDE up to a provided tolerance, given by the user. You can only play with the number of grid points `nx` and the number of time steps `nt`. For each experiment:
>
> 1. Call `solve_pde`.
> 2. Call `evaluate_solution` with the returned solution ID.
> 3. Use the maximum error to choose the next `nx` and `nt`.
>
> Before every call to `solve_pde`, briefly state your intention using exactly:
>
> **Action: _five words or fewer_**
>
> Examples:
>
> ```
> Action: Establish coarse baseline
> Action: Refine spatial resolution
> Action: Reduce timestep for stability
> Action: Approach error tolerance
> ```
>
> **Goal:** Achieve a maximum error < $10^{-3}$ with as little computation as possible.

Let's go through this prompt in more detail.

First, it is a **system prompt**. The language model sees this at every iteration, and it is always the first message in the conversation. API calls to the language model are stateless, so the runtime must resend the conversation history every time. Subsequent calls include both the model's previous responses and the results returned by tool calls.

The functions `solve_pde` and `evaluate_solution` are **tools** that the language model can invoke. These tools are exposed through a local **MCP server**. The language model never executes these functions itself—it merely requests that they be called. The runtime (our agent) performs the actual execution and feeds the results back into the conversation.

In this example, the model will end up calling both tools during every iteration because its goal is to improve the numerical solution. Nevertheless, it must still decide to do so.

Let's look at the runtime. The complete source code is available on [GitHub](https://github.com/hannesvdc/Poisson-Agent). At the beginning of every run, the runtime asks the MCP server which tools it exposes:

```python
mcp_tools = await mcp_client.list_tools()

tools = [
    ToolDefinition(
        name=tool.name,
        description=tool.description or "",
        input_schema=tool.inputSchema,
    )
    for tool in mcp_tools
]
```

In this example, the MCP server simply returns

```text
solve_pde(...)
evaluate_solution(...)
```

In a production-grade system it would probably make sense to merge these into a single `solve_and_evaluate` tool—we always intend to execute both together—but I found it useful to keep them separate while learning how agentic tool calling works.

Notice that the language model receives only the tool interface: the tool names, descriptions, and input schemas. The numerical implementation remains completely hidden behind the MCP server.

Next, we start the actual agent loop.

```python
while experiment < max_iterations:
    response = await provider.generate(
        messages=messages,
        tools=tools,
    )
```

This is the main interaction with the language model. The runtime is intentionally **provider agnostic**. The abstract `provider.generate()` interface can be implemented using the OpenAI SDK, Anthropic, or any other language model provider.

The model receives the complete conversation history together with the available tools and decides what experiment should be performed next. There are two possible outcomes. First, the model may determine that the task has already been completed because the most recent error satisfies the requested tolerance. In that case it simply returns a normal text response, the runtime performs no further tool calls, and the loop terminates. The second—and much more interesting—possibility is that the model decides to perform another numerical experiment. For example, it might produce

```text
Action: Establish coarse baseline

solve_pde(
    name="manufactured_sine",
    config={
        "nx": 20,
        "nt": 20
    }
)
```

This instructs the runtime to solve the PDE using $n_x=20$ spatial grid points and $n_t=20$ time steps. The name `manufactured_sine` is simply the identifier of the benchmark problem.

The runtime records this assistant response before executing the requested tool through the MCP server:

```python
result = await mcp_client.call_tool(
    call.name,
    call.arguments,
)
```

At this point the language model is no longer involved. The MCP server executes the finite-difference solver and returns a solution identifier.

The runtime then appends the tool result to the conversation history:

```python
messages.append(
    Message(
        role="tool",
        tool_call_id=call.id,
        content=result,
    )
)
```

This step is perhaps the most important—and also the easiest to overlook.

Appending previous assistant responses and tool outputs is how the agent observes the consequences of its own actions. The language model itself has no persistent memory. The runtime constructs that memory by replaying the conversation at every iteration.

On the next iteration, the language model receives not only the original user request, but also the outcome of its previous experiment. This feedback loop is the "secret sauce" that makes the agent work.

The first interaction therefore looks roughly like this:

```text
User:
Solve the manufactured_sine problem.

Assistant:
Action: Establish coarse baseline.
Call solve_pde(nx=20, nt=20).

Tool:
solution_id = "..."

Assistant:
Call evaluate_solution(solution_id="...").

Tool:
max_error = 8.1e20
CFL = 18.05
runtime = 0.001 s
```

The language model now has new information. The experiment failed catastrophically, and the timestep is far beyond the stability limit (CFL > 0.5). During the next iteration it may decide to increase the number of timesteps while keeping the spatial discretization fixed.

Notice that **none of this logic is hardcoded**.

There is no numerical rule such as

```python
if cfl > 0.5:
    increase_nt()
```

which would be typical in a traditional adaptive numerical algorithm.

Instead, the runtime simply implements the following feedback loop:

> **Reason → Act → Observe → Reason → Change Parameters**

The language model decides *how* to refine the discretization. The runtime itself is completely oblivious to numerical analysis. It knows nothing about stability conditions, finite-difference stencils, or PDEs. Its only responsibilities are to execute tools, return their outputs to the language model, and continue the loop until the model decides that the task has been completed.

## The Agent in Action

So let's see how well this pipeline works.

As a first example, we solve the heat equation (1) on the domain $x \in [0,1]$ with zero Dirichlet boundary conditions and the initial heat profile

$$
T_0(x)=T(x,0)=\sin(\pi x).
$$

The analytical solution is

$$
T(x,t)=T_0(x)\exp(-\alpha t),
$$

with diffusion constant $\alpha=1$.

At every iteration, the agent prints one row in the following optimization history.

| Iter | $n_x$ | $n_t$ | CFL | Max Error | Runtime | Action |
|------:|------:|------:|----:|----------:|--------:|--------|
| 1 | 20 | 20 | 18.050 | $2.560\times10^{20}$ | 0.001 s | Establish coarse baseline |
| 2 | 20 | 800 | 0.451 | $1.433\times10^{-3}$ | 0.022 s | Fix stability, increase timesteps |
| 3 | 25 | 1000 | 0.576 | $9.835\times10^{94}$ | 0.026 s | Increase resolution slightly further |
| 4 | 20 | 1600 | 0.226 | $2.956\times10^{-4}$ | 0.034 s | Increase timesteps only, keep $n_x$ |

The agent concludes by saying

```text
Achieved max_error = 2.96e-4 < 1e-3 tolerance, with modest computation
(nx=20, nt=1600, runtime ~0.034s). This satisfies the goal efficiently —
increasing nx destabilizes the CFL condition given fixed diffusivity,
so keeping nx=20 while raising nt was the right approach.

Final solution:
nx=20, nt=1600, max_error = 2.96e-4 (< 1e-3 tolerance).
```

Let's analyze what the agent has been doing.

Its initial coarse baseline of $n_x=n_t=20$ produces an unstable finite-difference solution. The error immediately explodes. The logical next step is to reduce the timestep size by increasing $n_t$, which is exactly what the agent does. Increasing $n_t$ to 800 stabilizes the simulation and reduces the error to $1.43\times10^{-3}$—already close to the requested tolerance.

At this point the agent reasons that 20 grid points are unlikely to be sufficient because the error depends on both $\Delta x$ and $\Delta t$. It therefore increases the spatial resolution to $n_x=25$. As expected, reducing $\Delta x$ increases the CFL number, making the explicit scheme unstable again. The agent then compensates by increasing the number of timesteps until the stability condition

$$
\frac{\alpha\Delta t}{\Delta x^2}
<
\frac12
$$

is satisfied once more.

This is exactly the sequence of refinements that a human numerical analyst would typically make when balancing stability and accuracy.

Interestingly, the reasoning process is **non-deterministic**. Running the same experiment a second time produces a different optimization history.

| Iter | $n_x$ | $n_t$ | CFL | Max Error | Runtime | Action |
|------:|------:|------:|----:|----------:|--------:|--------|
| 1 | 20 | 20 | 18.050 | $2.560\times10^{20}$ | 0.001 s | Establish coarse baseline |
| 2 | 20 | 800 | 0.451 | $1.433\times10^{-3}$ | 0.023 s | Increase timesteps for stability |
| 3 | 24 | 1000 | 0.529 | $1.509\times10^{27}$ | 0.026 s | Approach error tolerance |
| 4 | 20 | 1600 | 0.226 | $2.956\times10^{-4}$ | 0.033 s | Increase timesteps further |

The agent converges in the same number of iterations, but follows a different path.

Our second example introduces a sine-like forcing term

$$
f(x,t)
=
(4\alpha\pi^2-1)
\exp(-t)
\sin(2\pi x),
$$

with $x\in[0,1]$ and initial condition

$$
T_0(x)=\sin(2\pi x).
$$

Once again, the agent successfully solves the problem, producing the following optimization history.

| Iter | $n_x$ | $n_t$ | CFL | Max Error | Runtime | Action |
|------:|------:|------:|----:|----------:|--------:|--------|
| 1 | 20 | 20 | 18.050 | $8.112\times10^{20}$ | 0.001 s | Establish coarse baseline |
| 2 | 20 | 2000 | 0.181 | $8.297\times10^{-3}$ | 0.045 s | Reduce timestep for stability |
| 3 | 40 | 4000 | 0.380 | $1.965\times10^{-3}$ | 0.083 s | Increase resolution both dimensions |
| 4 | 60 | 9000 | 0.387 | $8.580\times10^{-4}$ | 0.127 s | Approach error tolerance |

The convergence story is again straightforward. The coarse baseline is unstable, prompting the agent to reduce the timestep. Once the simulation becomes stable, the remaining error is dominated by spatial discretization, leading the agent to refine the mesh and increase the number of timesteps in tandem. After four experiments, the desired accuracy has been reached.

The agent summarizes its work as follows:

```text
We achieved a max error of 8.58e-4, which is below the 1e-3 tolerance target.

Summary:
- Final configuration: nx=60, nt=9000
- Max error: 8.58e-4
- CFL number: 0.387 (stable)
- Runtime: 0.127 s

The solution converged efficiently: starting from an unstable coarse
baseline (CFL=18, error≈1e20), stabilizing the timestep brought the
error down to 8.3e-3, after which progressively refining both the
spatial and temporal resolution reduced the error below the requested
tolerance while maintaining stability.
```

## Some Thoughts
I really like how these experiments turned out. They show how AI agents can reason on the level of undergraduate students to solve numerical problems. The heat equation is far from the most complex PDE out there, but I believe this agent can handle much more. 

One idea I want to explore next is applying the same agentic loop to finite element methods, where the agent is responsible for proposing adaptive mesh refinements based on the observed error. Unlike the simple one-dimensional heat equation presented here, realistic engineering and scientific simulations often involve complex three-dimensional geometries, nonlinear physics, and millions of degrees of freedom. It would be fascinating to see how far an LLM can assist in designing efficient refinement strategies while leaving the underlying numerical computations to deterministic software.

Looking further ahead, I am even more interested in letting the agent design its own numerical algorithms. Instead of being given a fixed finite-difference solver, the agent could decide which discretization to use, write the numerical code, validate it against analytical solutions or manufactured problems, and iteratively improve both the algorithm and its implementation. The agent would then essentially mirror how computational scientists work!