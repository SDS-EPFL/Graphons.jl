```@meta
EditURL = "../../literate/tutorials/05_stochastic_shape_models.jl"
```

# Stochastic Shape Models

This tutorial introduces **Stochastic Shape Models (SSMs)**, a powerful extension of
Stochastic Block Models that reduces the number of parameters while maintaining the
flexibility to model complex network structures.

SSMs address a fundamental challenge in network modeling: standard SBMs with k blocks
require k² parameters (one for each block pair), which grows quickly. SSMs reduce this
by using a small number of "shapes" (connectivity patterns) that are shared across
multiple block pairs.

## What is a Stochastic Shape Model?

A **Stochastic Shape Model** is defined by three components:

1. **Shape parameters** θ: A vector of S edge probabilities (or distributions), where S << k²
2. **Shape assignment matrix** Ψ: A k×k matrix that assigns each block pair (i,j) to one of the S shapes
3. **Block sizes**: The proportion of nodes in each block (same as in SBMs)

The key insight: many block pairs can share the same connectivity pattern!

### Mathematical Formulation

For nodes i and j in blocks b(i) and b(j), the edge probability is:

```math
P(\text{edge}_{i,j}) = \theta_{\Psi(b(i), b(j))}
```

Instead of k² unique probabilities, we only need S parameters where S << k².

## Motivation: Parameter Efficiency

Consider a k=10 block network:
- **Standard SBM**: Requires 100 parameters (10×10 matrix)
- **SSM with 5 shapes**: Requires only 5 parameters + assignment matrix

This makes SSMs particularly useful for:
- **Large networks** with many blocks but limited data
- **Hierarchical structures** where similar patterns repeat
- **Model selection** with limited samples to avoid overfitting

## Setup

Load the packages we'll need:

````@example 05_stochastic_shape_models
using Graphons
using Random
using CairoMakie
using Distributions

Random.seed!(42)
````

## Example 1: Simple Bipartite Structure

Let's start with a simple example: a network with 3 blocks where we want:
- High connectivity within blocks (assortative)
- Low connectivity between blocks (sparse)

Using only **2 shapes** instead of 9 SBM parameters:

Define the shapes:

````@example 05_stochastic_shape_models
θ = [0.8,  # Shape 1: high probability (within-block)
    0.1]  # Shape 2: low probability (between-block)
````

Assign shapes to block pairs:

````@example 05_stochastic_shape_models
block_pair_to_shape = [1 2 2;   # Block 1: high within, low to others
                       2 1 2;   # Block 2: high within, low to others
                       2 2 1]   # Block 3: high within, low to others
````

Block sizes:

````@example 05_stochastic_shape_models
sizes = [0.3, 0.4, 0.3]
````

Create the SSM:

````@example 05_stochastic_shape_models
ssm_simple = SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))
````

Let's visualize both the shape assignment and a sampled graph:

````@example 05_stochastic_shape_models
fig = Figure(size = (900, 400))

ax1 = Axis(fig[1, 1],
    title = "Shape Assignment Matrix\n(which shape for each block pair)",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Equivalent Block Probability Matrix\n(θ values)",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax3 = Axis(fig[1, 3],
    title = "Sampled Graph (n=200)",
    aspect = 1)
````

Visualize shape assignments

````@example 05_stochastic_shape_models
heatmap!(ax1, block_pair_to_shape)
````

Visualize the equivalent probability matrix using direct plotting

````@example 05_stochastic_shape_models
hm = heatmap!(ax2, ssm_simple, colormap = :binary, colorrange = (0, 1))
````

Sample and visualize graph

````@example 05_stochastic_shape_models
hidedecorations!(ax3)
A_simple = sample_graph(ssm_simple, 200)
heatmap!(ax3, A_simple, colormap = :binary)

Colorbar(fig[1, 4], hm, label = "Edge probability")

fig
````

**Key observation**: We achieved the same result as a 3-block SBM but using only
2 parameters instead of 9! The shape assignment matrix shows that diagonal blocks
use shape 1 (high connectivity) while off-diagonal use shape 2 (low connectivity).

## Example 2: Hierarchical Core-Periphery

SSMs excel at modeling hierarchical structures. Let's create a 6-block network with:
- A core of 2 blocks (densely connected)
- A periphery of 4 blocks (sparsely connected)
- Medium connectivity between core and periphery

We'll use only **3 shapes**:

````@example 05_stochastic_shape_models
θ_hier = [0.9,  # Shape 1: dense (core-core)
    0.5,  # Shape 2: medium (core-periphery)
    0.1]  # Shape 3: sparse (periphery-periphery)
````

Create the shape assignment matrix for hierarchical structure:

````@example 05_stochastic_shape_models
block_pair_to_shape_hier = [1 1 2 2 2 2;   # Core block 1
                            1 1 2 2 2 2;   # Core block 2
                            2 2 3 3 3 3;   # Periphery block 1
                            2 2 3 3 3 3;   # Periphery block 2
                            2 2 3 3 3 3;   # Periphery block 3
                            2 2 3 3 3 3]

sizes_hier = [0.15, 0.15, 0.175, 0.175, 0.175, 0.175]  # Core smaller than periphery

ssm_hier = SSM(θ_hier, block_pair_to_shape_hier, sizes_hier, cumsum(sizes_hier))
````

Visualize the hierarchical structure:

````@example 05_stochastic_shape_models
fig = Figure(size = (1100, 400))

ax1 = Axis(fig[1, 1],
    title = "Shape Assignment Matrix\n(6 blocks, 3 shapes)",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Equivalent 6×6 Probability Matrix",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax3 = Axis(fig[1, 3],
    title = "Sampled Graph (n=300)",
    aspect = 1)

heatmap!(ax1, block_pair_to_shape_hier, colormap = :tab10, colorrange = (1, 3))

hm = heatmap!(ax2, ssm_hier, colormap = :binary, colorrange = (0, 1))

hidedecorations!(ax3)
A_hier = sample_graph(ssm_hier, 300)
heatmap!(ax3, A_hier, colormap = :binary)

Colorbar(fig[1, 4], hm, label = "Edge probability")

fig
````

**Key advantage**: This 6-block hierarchical model uses only 3 parameters instead of 36!
The clear two-level hierarchy (dense core in top-left, sparse periphery in bottom-right)
is efficiently captured by shape sharing.

## Example 3: Modular Networks with Weak Ties

Real networks often have multiple dense communities connected by sparse "weak ties".
Let's model a 4-community network where:
- Communities 1-2 form one meta-module
- Communities 3-4 form another meta-module
- Within-module connections are stronger than between-module

Using **4 shapes**:

````@example 05_stochastic_shape_models
θ_modular = [0.9,  # Shape 1: within-community
    0.6,  # Shape 2: within-module, between-community
    0.2,  # Shape 3: between-module
    0.05] # Shape 4: no connection (rare edges)

block_pair_to_shape_modular = [1 2 3 4;   # Community 1
                               2 1 4 3;   # Community 2
                               3 4 1 2;   # Community 3
                               4 3 2 1]

sizes_modular = [0.25, 0.25, 0.25, 0.25]

ssm_modular = SSM(θ_modular, block_pair_to_shape_modular,
    sizes_modular, cumsum(sizes_modular))
````

Visualize:

````@example 05_stochastic_shape_models
fig = Figure(size = (1100, 400))

ax1 = Axis(fig[1, 1],
    title = "Shape Assignment\n(4 communities, 4 shapes)",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Probability Matrix",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax3 = Axis(fig[1, 3],
    title = "Sampled Graph (n=200)",
    aspect = 1)

heatmap!(ax1, block_pair_to_shape_modular, colormap = :tab10, colorrange = (1, 4))

hm = heatmap!(ax2, ssm_modular, colormap = :binary, colorrange = (0, 1))

hidedecorations!(ax3)
A_modular = sample_graph(ssm_modular, 200)
heatmap!(ax3, A_modular, colormap = :binary)

Colorbar(fig[1, 4], hm, label = "Edge probability")

fig
````

Notice the symmetric checkerboard pattern! This reflects the two meta-modules.
The SSM uses only 4 parameters to capture this complex structure, compared to
16 parameters for an equivalent SBM.

## Converting SSMs to SBMs

Every SSM can be converted to an equivalent SBM by expanding the shape assignments:

````@example 05_stochastic_shape_models
sbm_from_ssm = Graphons.convert_to_sbm(ssm_modular)

println("SSM type: ", typeof(ssm_modular))
println("SSM parameters: ", length(ssm_modular.θ), " shapes")
println("\nConverted SBM type: ", typeof(sbm_from_ssm))
println("SBM matrix size: ", size(sbm_from_ssm.θ))
println("SBM parameters: ", length(sbm_from_ssm.θ), " entries")
````

The SSM and SBM are mathematically equivalent, but the SSM representation
is more **compact** and **interpretable** when many block pairs share patterns.

## Decorated SSMs: Adding Edge Attributes

Just like SBMs can be decorated with distributions, SSMs can too!
This combines parameter efficiency with rich edge attributes.

Let's create an SSM with weighted edges:

Define shape distributions (instead of probabilities):

````@example 05_stochastic_shape_models
θ_weighted = [
    Normal(5.0, 0.5),   # Shape 1: strong positive connections
    Normal(0.0, 0.3),   # Shape 2: weak/zero connections
    Normal(-2.0, 0.5)   # Shape 3: negative/inhibitory connections
]
````

3-block network with signed edges:

````@example 05_stochastic_shape_models
block_pair_to_shape_weighted = [1 2 3;   # Block 1: positive within, weak to 2, negative to 3
                                2 1 2;   # Block 2: positive within, weak elsewhere
                                3 2 1]

sizes_weighted = [0.35, 0.3, 0.35]

ssm_weighted = SSM(θ_weighted, block_pair_to_shape_weighted,
    sizes_weighted, cumsum(sizes_weighted))
````

Sample a weighted graph:

````@example 05_stochastic_shape_models
A_weighted = sample_graph(ssm_weighted, 150)

println("Weighted matrix type: ", typeof(A_weighted))
println("Matrix size: ", size(A_weighted))
println("Value range: [", minimum(A_weighted), ", ", maximum(A_weighted), "]")
````

Visualize the weighted network:

````@example 05_stochastic_shape_models
fig = Figure(size = (1100, 400))

ax1 = Axis(fig[1, 1],
    title = "Shape Assignment",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Shape Distributions\n(means)",
    xlabel = "Block",
    ylabel = "Block",
    aspect = 1)

ax3 = Axis(fig[1, 3],
    title = "Sampled Weighted Graph (n=150)",
    aspect = 1)

heatmap!(ax1, block_pair_to_shape_weighted, colormap = :tab10, colorrange = (1, 3))
````

Show mean of each distribution

````@example 05_stochastic_shape_models
θ_means = [mean(d) for d in θ_weighted]
θ_matrix_weighted = [θ_means[block_pair_to_shape_weighted[i, j]]
                     for i in 1:3, j in 1:3]
hm_means = heatmap!(ax2, θ_matrix_weighted, colormap = :RdBu,
    colorrange = (-3, 6))

hidedecorations!(ax3)
hm_weights = heatmap!(ax3, A_weighted, colormap = :RdBu,
    colorrange = (-4, 7))

Colorbar(fig[1, 4], hm_weights, label = "Edge weight")

fig
````

**Interpretation**: The SSM creates three distinct types of connections:
- Diagonal (shape 1): Strong positive weights (red)
- Off-diagonal corners (shape 3): Negative weights (blue)
- Other pairs (shape 2): Weak weights (white)

## Example 4: Degree-Corrected SSM

We can create more realistic networks by varying block sizes to simulate
degree heterogeneity while keeping the shape structure:

Same shapes as before, but with very unequal block sizes:

````@example 05_stochastic_shape_models
θ_dc = [0.9, 0.3, 0.05]
block_pair_to_shape_dc = [1 2 3;
                          2 1 2;
                          3 2 1]
sizes_dc = [0.6, 0.3, 0.1]  # Highly unequal blocks

ssm_dc = SSM(θ_dc, block_pair_to_shape_dc, sizes_dc, cumsum(sizes_dc))
````

Compare equal vs unequal sizes:

````@example 05_stochastic_shape_models
fig = Figure(size = (900, 400))

ax1 = Axis(fig[1, 1],
    title = "Equal Block Sizes\n[0.33, 0.33, 0.33]",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Unequal Block Sizes\n[0.6, 0.3, 0.1]",
    aspect = 1)

hidedecorations!.([ax1, ax2])
````

Sample with equal sizes

````@example 05_stochastic_shape_models
ssm_equal = SSM(θ_dc, block_pair_to_shape_dc, [0.33, 0.33, 0.34],
    cumsum([0.33, 0.33, 0.34]))
A_equal = sample_graph(ssm_equal, 300)
heatmap!(ax1, A_equal, colormap = :binary)
````

Sample with unequal sizes

````@example 05_stochastic_shape_models
A_unequal = sample_graph(ssm_dc, 300)
heatmap!(ax2, A_unequal, colormap = :binary)

fig
````

## Analyzing Shape Assignments

Let's analyze which block pairs are assigned to which shapes:

````@example 05_stochastic_shape_models
function analyze_shape_assignments(ssm)
    K = length(ssm.size)
    S = length(ssm.θ)

    println("Number of blocks: ", K)
    println("Number of shapes: ", S)
    println("\nShape assignments:")

    for s in 1:S
        pairs = [(i, j) for i in 1:K for j in 1:K
                 if ssm.block_pair_to_shape[i, j] == s]
        n_pairs = length(pairs)
        println("\nShape $s (θ = $(ssm.θ[s])):")
        println("  Used by $n_pairs block pairs ($(round(100*n_pairs/K^2, digits=1))%)")
        if n_pairs ≤ 6
            println("  Pairs: ", pairs)
        end
    end
end

analyze_shape_assignments(ssm_modular)
````

This helps understand how parameter sharing works in the model.

## Comparison with Standard SBM

Let's compare the same network structure using both SSM and SBM:

SSM version (4 shapes):

````@example 05_stochastic_shape_models
ssm_compare = ssm_modular
````

Equivalent SBM (16 parameters):

````@example 05_stochastic_shape_models
sbm_compare = Graphons.convert_to_sbm(ssm_compare)
````

Sample from both:

````@example 05_stochastic_shape_models
Random.seed!(123)
A_ssm = sample_graph(ssm_compare, 200)

Random.seed!(123)
A_sbm = sample_graph(sbm_compare, 200)
````

Verify they're identical:

````@example 05_stochastic_shape_models
println("SSM and SBM produce identical graphs: ", A_ssm == A_sbm)

fig = Figure(size = (900, 400))

ax1 = Axis(fig[1, 1],
    title = "SSM (4 parameters)\nShapes: [0.9, 0.6, 0.2, 0.05]",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Equivalent SBM (16 parameters)\n4×4 full matrix",
    aspect = 1)

hidedecorations!.([ax1, ax2])
heatmap!(ax1, A_ssm, colormap = :binary)
heatmap!(ax2, A_sbm, colormap = :binary)

fig
````

**Identical networks**, but SSM uses 75% fewer parameters!

## Advanced: Random Shape Assignments

For exploratory modeling, we can create SSMs with random shape assignments
to see what structures emerge:

````@example 05_stochastic_shape_models
function random_ssm(K, S, rng = Random.GLOBAL_RNG)
    θ = sort(rand(rng, S), rev = true)

    assignments = rand(rng, 1:S, K, K)
    block_pair_to_shape = (assignments + assignments') .÷ 2

    sizes = rand(rng, K)
    sizes ./= sum(sizes)

    return SSM(θ, block_pair_to_shape, sizes, cumsum(sizes))
end

Random.seed!(789)
ssm_random = random_ssm(6, 3)

A_random = sample_graph(ssm_random, 180)

fig = Figure(size = (1100, 400))

ax1 = Axis(fig[1, 1],
    title = "Random Shape Assignment",
    aspect = 1)

ax2 = Axis(fig[1, 2],
    title = "Equivalent Probability Matrix",
    aspect = 1)

ax3 = Axis(fig[1, 3],
    title = "Sampled Random SSM",
    aspect = 1)

heatmap!(ax1, ssm_random.block_pair_to_shape, colormap = :tab10)
hm = heatmap!(ax2, ssm_random, colormap = :binary, colorrange = (0, 1))
hidedecorations!(ax3)
heatmap!(ax3, A_random, colormap = :binary)

Colorbar(fig[1, 4], hm, label = "Edge probability")

fig
````

Random SSMs can generate surprisingly diverse structures!

## Key Takeaways

- **Stochastic Shape Models** reduce parameters by sharing connectivity patterns (shapes) across block pairs
- An SSM with k blocks and S shapes uses only S parameters instead of k² parameters
- Shape assignments are encoded in a k×k matrix Ψ that maps block pairs to shapes
- SSMs are ideal for:
  - Large networks with many blocks but limited data
  - Hierarchical or modular structures with repeating patterns
  - Parsimonious, interpretable models
- Every SSM can be converted to an equivalent SBM using `convert_to_sbm()`
- Decorated SSMs combine parameter efficiency with rich edge attributes (weights, distributions)
- Use `get_theta_matrix()` to visualize the full probability matrix from shapes

## References

Stochastic Shape Models were introduced in [verdeyme_hybrid_2024](@cite)


The paper provides:
- Theoretical properties (consistency, rates of convergence)
- Model selection procedures for choosing S
- Applications to real network data
- Comparisons with standard SBMs and other graphon models

## Real-World Example: Multiplex Network from Research

Let's recreate an example from recent research on decorated graphon estimation
(Dufour & Olhede, 2024). This example demonstrates how SSMs can provide smoother
approximations than standard SBMs for complex multiplex networks.

The graphon `w3` models a 4-category multiplex network where edge probabilities
are determined by a softmax transformation of four distinct spatial patterns:

````@example 05_stochastic_shape_models
using LogExpFunctions: softmax!
function w3(x, y)
    tabulation = zeros(4)
    tabulation[1] = 3 * x * y                                      # Linear interaction
    tabulation[2] = 3 * sin(2 * π * x) * sin(2 * π * y)           # Periodic pattern
    tabulation[3] = exp(-3 * ((x - 0.5)^2 + (y - 0.5)^2))        # Gaussian bump
    tabulation[4] = 2 - 3 * (x + y)                               # Decreasing pattern
    softmax!(tabulation)
    return DiscreteNonParametric(0:3, tabulation)
end
````

Create the decorated graphon:

````@example 05_stochastic_shape_models
graphon_w3 = DecoratedGraphon(w3)
````

Visualize the four category probabilities:

````@example 05_stochastic_shape_models
fig = Figure(size = (1000, 250))

for i in 1:4
    ax = Axis(fig[1, i],
        title = "Category $i",
        aspect = 1)
    hidedecorations!(ax)
    hm = heatmap!(ax, graphon_w3, k = i, colormap = :binary, colorrange = (0, 1))
end

Colorbar(fig[1, 5], colormap = :binary, colorrange = (0, 1),
    label = "Probability", height = Relative(0.8))

fig
````

### Comparing SBM vs SSM Approximations

Create a standard SBM approximation with k=10 blocks:

````@example 05_stochastic_shape_models
k_blocks = 15
sbm_w3 = empirical_graphon(graphon_w3, k_blocks)
````

For the SSM, we'll use fewer shapes (s=30) than the SBM parameters (k(k+1)/2=120):
This demonstrates the parameter efficiency of SSMs.
In the `Graphons` package, to automatically create an SSM from an SBM, we can load the
`Clustering` package and use the `SSM` constructor:

````@example 05_stochastic_shape_models
using Clustering

ssm_w3 = SSM(sbm_w3, 30)
````

We can now visualize the difference between the SBM and SSM approximations:

````@example 05_stochastic_shape_models
fig = Figure(size = (1000, 500))

for i in 1:4
    ax = Axis(fig[1, i],
        title = "Category $i",
        aspect = 1)
    hidedecorations!(ax)
    hm = heatmap!(ax, sbm_w3, k = i, colormap = :binary, colorrange = (0, 1))
    ax2 = Axis(fig[2, i],
        aspect = 1)
    hidedecorations!(ax2)
    hm = heatmap!(ax2, ssm_w3, k = i, colormap = :binary, colorrange = (0, 1))
end

Colorbar(fig[1, 5], colormap = :binary, colorrange = (0, 1),
    label = "Probability", height = Relative(0.8))

fig
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

