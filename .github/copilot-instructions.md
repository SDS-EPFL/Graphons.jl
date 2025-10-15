# Copilot Instructions for Graphons.jl

## Project Overview

Graphons.jl is a Julia package for sampling random graphs from graphon models.
The package supports simple graphs, weighted graphs, and decorated graphs (with
distributions on edges) using a flexible type system.

## Core Architecture

### Type Hierarchy

The central abstraction is `AbstractGraphon{T,M}` where:

- `T` is the edge type: `Bool` (simple), `<:Real` (weighted), or
  `<:AbstractVector{Bool}` (multiplex)
- `M` is the graph representation: `BitMatrix`, `Matrix{T}`, `SparseMatrixCSC`,
  or `GBMatrix` (via extension)

Key types in `src/`:

- `SimpleContinuousGraphon`: Continuous graphon functions
  `f(x,y) -> probability`
- `SBM`: Stochastic Block Models with discrete blocks
- `DecoratedGraphon`: Returns distributions instead of probabilities
- `DecoratedSBM`: Block model variant with distributions

### File Organization

- `src/Graphons.jl`: Main module with type definitions and exports
- `src/simple_graphon.jl`: Simple/SBM implementations
- `src/decorated_graphon.jl`: Distribution-based graphons
- `src/sampling.jl`: `rand()` and `sample_graph()` implementations
- `src/utils.jl`: Graph utilities (`make_empty_graph`, `clear_graph!`,
  `_convert_latent_to_block`)
- `ext/`: Optional extensions for Makie (plotting) and SuiteSparseGraphBLAS
  (sparse matrices)

## Architectural Decisions & Design Rationale

### Why Two Type Parameters {T,M}?

The `AbstractGraphon{T,M}` design separates **what** edges represent (T) from
**how** graphs are stored (M):

- **Flexibility**: Users can choose storage backends (dense, sparse, GPU)
  without changing graphon logic
- **Performance**: Sparse matrices for low-density graphs, dense for
  high-density, GPU matrices for large-scale sampling
- **Type Stability**: Julia compiler can specialize `_rand!` for each (T,M)
  pair, eliminating runtime dispatch

Example: Same SBM can generate `BitMatrix` for memory efficiency or
`GBMatrix{Bool}` for GraphBLAS algorithms, just by changing M.

### Why Separate Simple vs Decorated Graphons?

**SimpleContinuousGraphon** and **DecoratedGraphon** have different sampling
semantics:

- **Simple**: `f(x,y) -> Float64` is a probability. Sample edge with
  `rand() < f(x,y)`
- **Decorated**: `f(x,y) -> Distribution`. Sample edge value with
  `rand(distribution)`

This split enables:

1. **Optimized sampling**: Simple graphons use fast Bernoulli trials without
   allocation
2. **Rich edge attributes**: Decorated graphons support any Distribution.jl
   type (Poisson for counts, Normal for weights, Categorical for labels)
3. **Type inference**: `DecoratedGraphon(f)` probes `f(0.1, 0.2)` to infer
   return type, avoiding manual type annotations

### Why Both rand() and sample_graph()?

Two sampling functions serve different use cases:

- **`rand(graphon, n)`**: Quick random graphs for Monte Carlo, exploratory
  analysis. Latents drawn uniform [0,1], convenient default.
- **`sample_graph(graphon, ξs)`**: Controlled sampling with fixed latents.
  Critical for:
  - Reproducible experiments (same ξs = same graph structure)
  - Conditional sampling (fix some nodes, sample others)
  - Theoretical analysis (relate graph properties to latent positions)

Implementation: Both call `_rand!(rng, f, A, ξs)` internally. The distinction
is purely API-level convenience.

### Extension System Design

Extensions (`ext/`) avoid hard dependencies while supporting specialized
backends:

- **Makie**: Plotting is optional (not everyone needs visualization)
- **SuiteSparseGraphBLAS**: GraphBLAS matrices for sparse linear algebra (large
  graphs)

Each extension implements the **same interface** (`make_empty_graph`,
`clear_graph!`) so core code remains backend-agnostic. Extensions load
automatically when users install weak dependencies.

**Why this matters**: Package stays lightweight (<10 dependencies) while
supporting power users with GPU/sparse backends.

### Why Latent Space on [0,1]?

All graphon models use latent variables ξ ∈ [0,1]:

- **Mathematical convenience**: Standard measure theory assumptions, canonical
  formulation
- **Block model mapping**: `_convert_latent_to_block` uses cumulative sizes to
  partition [0,1] into blocks
- **Uniform semantics**: `rand()` draws uniform latents; `sample_graph(n)` uses
  `range(0,1,length=n)` for deterministic spacing

Alternative designs (arbitrary domains, multi-dimensional latents) would break
the clean SBM ↔ continuous graphon relationship.

### Symmetry by Construction

`_rand!` fills only upper triangle, then mirrors to lower. Design rationale:

- **Undirected graphs**: Most network models are symmetric (social networks,
  co-authorship)
- **Storage efficiency**: No need to store both A[i,j] and A[j,i]
- **Consistency**: Impossible to accidentally create asymmetric graphs

Diagonal is always zero (no self-loops) following standard network convention.

## Critical Patterns

### Sampling Functions

Two sampling approaches exist:

1. `rand(rng, graphon, n)`: Random latent positions (uniform on [0,1])
2. `sample_graph(rng, graphon, ξs)`: Fixed latent positions

Both delegate to `_rand!(rng, f, A, ξs)` which:

- Fills upper triangle only (no diagonal)
- Mirrors to lower triangle for symmetry
- Has specialized implementations for `SimpleGraphon` (probability threshold)
  vs general (distribution sampling)

### Type Inference for Decorated Graphons

`DecoratedGraphon` constructor calls `f(0.1, 0.2)` to infer types:

- `_infer_eltype(d::Distribution)` returns `eltype(d)`
- `_infer_eltype(d::MultivariateDistribution)` wraps in `SizedVector`
- This enables type stability without explicit type parameters

### Validation Pattern

Use `@argcheck` from ArgCheck.jl for constructor validation:

```julia
@argcheck all(x -> x <= 1 && x >= 0, θ)  # Probabilities in [0,1]
@argcheck last(cumsizes) ≈ 1              # Sizes sum to 1
```

### Empirical Graphon Discretization

`empirical_graphon(f, k)` discretizes continuous graphons:

- Creates uniform grid of `k` points
- **Important**: Uses `sizes[end] += 1 - sum(sizes)` to ensure exact sum=1
  (floating point correction)

## Testing Structure

Tests are split by functionality in `test/`:

- `test_utils.jl`: Core utilities
- `test_simple_graphon.jl`, `test_sbm.jl`: Simple graph types
- `test_decorated_graphon.jl`, `test_decorated_sbm.jl`: Distribution-based
  types
- `test_sampling_rand.jl`, `test_sampling_sample_graph.jl`: Sampling functions
- `test_integration.jl`: End-to-end workflows
- `test_graphblas_ext.jl`: Extension tests (conditional on package
  availability)

Extension tests use conditional loading:

```julia
const PKG_AVAILABLE = try; using Package; true; catch; false; end
if PKG_AVAILABLE
    # tests
else
    @test_skip true "Package not available"
end
```

## Development Workflows

### Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Building Documentation

Documentation uses Literate.jl to convert `.jl` files to markdown:

```bash
julia --project=docs/ docs/make.jl
```

- Literate sources: `docs/literate/tutorials/*.jl`
- Generated output: `docs/src/tutorials/*.md`

### Adding Extensions

Extensions in `ext/` are loaded automatically when weak dependencies are
available:

1. Add to `[weakdeps]` in Project.toml
2. Create `ext/PackageExt.jl` module
3. Extend core functions: `make_empty_graph`, `clear_graph!`, `_rand!`
4. Add conditional tests to `test_graphblas_ext.jl` pattern

## Common Pitfalls

### Matrix Type Consistency

When creating graphons with custom matrix types, pass type as second argument:

```julia
SimpleContinuousGraphon(f, GBMatrix{Bool})  # ✓ Correct
SimpleContinuousGraphon(f) |> rand(_, 10)   # ✗ Uses default BitMatrix
```

### Symmetry and Diagonal

Sampling functions enforce:

- Symmetric adjacency: `A[i,j] == A[j,i]`
- No self-loops: `A[i,i] == 0`
- Use `issymmetric(Matrix(A))` for sparse/GBMatrix types

### Test Dependencies

`LinearAlgebra` must be in `[extras]` for `issymmetric()`, `diag()` in tests.

## Key Implementation Details

### Block Assignment

`_convert_latent_to_block(sbm, ξ)` uses `findfirst(y -> ξ <= y, sbm.cumsize)`
to map continuous positions to discrete blocks.

### Callable Graphons

All graphon types implement `(g::Graphon)(x, y)` to evaluate
probability/distribution at position `(x,y)`.

### Random Seeding

Tests use `MersenneTwister(seed)` for reproducibility. Always pass explicit RNG
to sampling functions in tests.
