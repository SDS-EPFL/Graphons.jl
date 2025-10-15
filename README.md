# Graphons.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SDS-EPFL.github.io/Graphons.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SDS-EPFL.github.io/Graphons.jl/dev/)
[![Build Status](https://github.com/SDS-EPFL/Graphons.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SDS-EPFL/Graphons.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SDS-EPFL/Graphons.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SDS-EPFL/Graphons.jl)

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

## Installation

```julia
using Pkg
Pkg.add("Graphons")
```

## Documentation

For more details and tutorials please refer to the documentation.
