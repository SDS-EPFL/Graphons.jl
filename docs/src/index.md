```@meta
CurrentModule = Graphons
```

# Graphons.jl

Documentation for [Graphons.jl](https://github.com/SDS-EPFL/Graphons.jl) - A
Julia package for sampling random graphs from graphon models.

## Overview

Graphons are infinite-dimensional objects that represent the limit of large
graphs. This package provides tools to:

- Define graphon models (continuous functions or block models)
- Sample finite graphs from these models
- Work with decorated graphons that have rich edge attributes

## Quick Start

```julia
using Graphons

# Create a simple continuous graphon
g = SimpleContinuousGraphon((x, y) -> 0.3)

# Sample a random graph with 100 nodes
A = rand(g, 100)

# Create a stochastic block model
θ = [0.8 0.1; 0.1 0.8]  # High within-block, low between-block probability
sizes = [0.5, 0.5]       # Equal-sized blocks
sbm = SBM(θ, sizes)
A = rand(sbm, 200)
```

## Features

- **Simple Graphons**: Work with continuous probability functions on [0,1]²
- **Stochastic Block Models**: Discrete graphons with block structure
- **Decorated Graphons**: Rich edge attributes using Distributions.jl or custom
  distributions
- **Extensible**: Easily add custom distribution types with just 2 methods
- **Flexible Storage**: Support for dense, sparse, and GraphBLAS matrices
- **Type Stability**: Optimized performance through Julia's type system

## Installation

```julia
using Pkg
Pkg.add("Graphons")
```

## Package Structure

The package exports:

- **Types**: `SimpleContinuousGraphon`, `SBM`, `DecoratedGraphon`,
  `DecoratedSBM`
- **Functions**: `rand`, `sample_graph`, `empirical_graphon`

See the [API Reference](@ref) for detailed documentation.

## Contents

```@contents
Pages = ["api.md", "graphs_type.md", "tutorials.md"]
Depth = 2
```
