```@meta
EditURL = "../../literate/tutorials/02_multiplex_networks.jl"
```

# Multiplex networks as finitely-decorated graphons (2 layers)

This tutorial shows how a 2-layer multiplex network can be expressed as a
finitely-decorated graphon with four categories `K = {0,1,2,3}` corresponding to:

* `0` = no edge on either layer (00)
* `1` = edge on layer 1 only (10)
* `2` = edge on layer 2 only (01)
* `3` = edge on both layers (11)

We'll
1. define a smooth decorated graphon `W(x,y)` returning a 4-probability vector over these categories,
2. sample a multiplex network with two binary layers,
3. visualise marginals and cross-layer correlation induced by `W`.

## Setup

````@example 02_multiplex_networks
using Random
using Distributions
using StaticArrays
using SparseArrays
using Graphons
using CairoMakie
````

## A 4-category decorated graphon for a 2-layer multiplex
We define a function $W = (w^{00},w^{10},w^{01},w^{11})$ that returns  a distribution on `K = {0,1,2,3}`

with $\mathbb{P}(K = k | X = x, Y = y) = w^{(k)}(x,y)$

````@example 02_multiplex_networks
function W(x, y)
    ps = zeros(4)
    ps[2] = sqrt(abs(x - y)) / 2
    ps[3] = abs(sin(2π * x) * sin(2π * y)) / 2
    ps[4] = min(x, y) / 4
    ps[1] = 1 - sum(ps[2:4])
    return DiscreteNonParametric(0:3, SVector{4}(ps))
end

g = DecoratedGraphon(W);
nothing #hide
````

## Visualising the four category probabilities

We plot the four surfaces $w^{(ℓ)}(x,y)$ for ℓ ∈ {0,1,2,3}.

````@example 02_multiplex_networks
fig = Figure(size=(350, 300)) # hide
ax1 = Axis(fig[1, 1], title="p00(x,y)", aspect=1) # hide
ax2 = Axis(fig[1, 2], title="p10(x,y)", aspect=1) # hide
ax3 = Axis(fig[2, 1], title="p01(x,y)", aspect=1) # hide
ax4 = Axis(fig[2, 2], title="p11(x,y)", aspect=1) # hide
xlims!.([ax1, ax2, ax3, ax4], 0, 1) # hide
ylims!.([ax1, ax2, ax3, ax4], 0, 1) # hide
hidexdecorations!.([ax1, ax2, ax3, ax4]) # hide
hideydecorations!.([ax1, ax2, ax3, ax4]) # hide
heatmap!(ax1, g, 1, colormap=:binary, colorrange=(0, 1)) # hide
heatmap!(ax2, g, 2, colormap=:binary, colorrange=(0, 1)) # hide
heatmap!(ax3, g, 3, colormap=:binary, colorrange=(0, 1)) # hide
heatmap!(ax4, g, 4, colormap=:binary, colorrange=(0, 1)) # hide
Colorbar(fig[:, 3], colormap=:binary, colorrange=(0, 1)) # hide
resize_to_layout!(fig)# hide
fig# hide
````

## Sampling a multiplex network from W

Given `W`, we can sample an adjacency *category* for each unordered pair (i,j)
and then split categories into two binary adjacency matrices `A1` and `A2`.

````@example 02_multiplex_networks
n = 300
Random.seed!(42)
A = sample_graph(g, n);
nothing #hide
````

## Visualising the categories

````@example 02_multiplex_networks
fig = Figure(size=(300, 300)) #hide
ax1 = Axis(fig[1, 1], aspect=1) # hide
ax2 = Axis(fig[1, 2], aspect=1) # hide
ax3 = Axis(fig[2, 1], aspect=1) # hide
ax4 = Axis(fig[2, 2], aspect=1) # hide
hidedecorations!.([ax1, ax2, ax3, ax4]) # hide
A_inter = zeros(Bool, n, n) # hide
for (c, ax) in zip(0:3, (ax1, ax2, ax3, ax4)) # hide
    indices = findall(x -> x == c, A)  # indices with category c # hide
    A_inter .= false # hide
    A_inter[indices] .= true # hide
    heatmap!(ax, A_inter, colormap=:binary, colorrange=(0, 1)) # hide
end # hide
resize_to_layout!(fig) # hide
fig # hide
````

## Visualising the sampled layers

````@example 02_multiplex_networks
indices_layer1 = findall(x -> x in (1, 3), A)  # hide
indices_layer2 = findall(x -> x in (2, 3), A)  # hide

A1 = zeros(Bool, n, n) # hide
A2 = zeros(Bool, n, n) # hide
A1[indices_layer1] .= true # hide
A2[indices_layer2] .= true # hide



fig = Figure(size=(300, 200)) # hide
ax1 = Axis(fig[1, 1], title="Layer 1", aspect=1) # hide
ax2 = Axis(fig[1, 2], title="Layer 2", aspect=1) # hide
hidedecorations!.([ax1, ax2]) # hide
heatmap!(ax1, A1, colormap=:binary, colorrange=(0, 1)) # hide
heatmap!(ax2, A2, colormap=:binary, colorrange=(0, 1)) # hide
resize_to_layout!(fig) # hide
fig # hide
````

## From categories to marginals and correlation

For the two Bernoulli layers, we can estimate, for each (i,j), the *model*
marginals and correlation induced by `W`. In practice we only observe one draw,
but for visualisation we map $(x,y)\rightarrow (p_1,p_2,corr)$ over a grid.

````@example 02_multiplex_networks
using MVBernoulli

function w_mvbern(x, y)
    MVBernoulli.from_tabulation([probs(W(x, y))...])
end

g_mvbern = DecoratedGraphon(w_mvbern);

sbm = empirical_graphon(g_mvbern, 10);


get_marginals(d, k) = marginals(d)[k]
get_correlation(d) = correlation_matrix(d)[1, 2]


p1 = [get_marginals(g_mvbern(x, y), 1) for x in 0:0.01:1, y in 0:0.01:1]
p2 = [get_marginals(g_mvbern(x, y), 2) for x in 0:0.01:1, y in 0:0.01:1]
corr = [get_correlation(g_mvbern(x, y)) for x in 0:0.01:1, y in 0:0.01:1]

#

fig = Figure(size=(800, 350)) # hide
ax1 = Axis(fig[1, 1], title="p1(x,y)", aspect=1) # hide
ax2 = Axis(fig[1, 2], title="p2(x,y)", aspect=1) # hide
ax3 = Axis(fig[1, 3], title="corr(x,y)", aspect=1) # hide
hidexdecorations!.([ax1, ax2, ax3]) # hide
hideydecorations!.([ax1, ax2, ax3]) # hide
hm1 = heatmap!(ax1, p1, colormap=:binary, colorrange=(0, 1)) # hide
heatmap!(ax2, p2, colormap=:binary, colorrange=(0, 1)) # hide
hm3 = heatmap!(ax3, corr, colormap=:balance, colorrange=(-1, 1)) # hide
Colorbar(fig[2, 1:2], hm1, vertical=false, width=Relative(0.5)) # hide
Colorbar(fig[2, 3], hm3, vertical=false) # hide
fig # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

