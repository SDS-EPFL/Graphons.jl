```@meta
EditURL = "../../literate/tutorials/01_simple_graphon.jl"
```

#  A Simple Graphon Introduction

This tutorial introduces the concept of a graphon, demonstrates how to sample a graph from one using `Graphon.jl`.

## What is a Graphon?

A graphon (or graph function) is a symmetric, measurable function $$W: [0, 1]^2 \to [0, 1]$$.
A graph with $n$ nodes is then generated in the following manner:
 For each node, $i$ a latent variable $\xi_i \sim U[0,1]$ is drawn independently of the others

It serves as a generative model for random graphs. Think of it as a continuous and more general version of a stochastic block model.

In simple terms, each node `i` in a graph is assigned a latent (unobserved) position $ξ_i \in [0, 1]$. The probability of an edge existing between two nodes `i` and `j` is then given by the graphon function evaluated at their latent positions: $$P[A_{ij} = 1 \mid \xi_i,\xi_j] = W(\xi_i,\xi_j).$$

This is an example of a simple graphon, which is used to generate simple binary undirected graphs. In subsequent tutorials, we will show that we can generalise this idea to much more general kind of graphs (weighted, signed, multiplex, temporal,...).

## Using `Graphon.jl` to deal with Graphon

We will be interested in a common graphon encountered in the litterature: $W(x,y)=x*y$. This graphon is a `Graphon.`

````@example 01_simple_graphon
using Graphons

function W(x, y)
    return x * y
end

f = SimpleContinuousGraphon(W);
nothing #hide
````

now that we have defined our graphon, we can sample graphs of different sizes with it:

````@example 01_simple_graphon
using Random
A_medium = rand(f, 11)
````

The above call will generate 10 random latent variables, and then sample the graph according to these random latents.
In some settings we might be interested in knowing the latents for each of the nodes (e.g. for simulations). This is also possible easily:

````@example 01_simple_graphon
ξs = 0:0.1:1
A_ordered = sample(f, ξs)
````

### Specifying the type of the sampled graph

````@example 01_simple_graphon
using SparseArrays
M = SparseMatrixCSC{Bool,Int}
f_sparse = SimpleContinuousGraphon(W, M)
rand(f_sparse, 41)
````

and we can see the impact of ordering the latents

````@example 01_simple_graphon
sample(f_sparse, 0:0.025:1)
````

We can also use the `SuiteSparseGraphBLAS` package to sample large sparse graphs efficiently:

````@example 01_simple_graphon
using SuiteSparseGraphBLAS
M2 = GBMatrix{Bool,Bool}
f_sparseBLAS = SimpleContinuousGraphon(W, M2)
rand(f_sparseBLAS, 41);
nothing #hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

