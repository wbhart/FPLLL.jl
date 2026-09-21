# FPLLL.jl

A small Julia interface to the LLL functionality in
[fpLLL](https://github.com/fplll/fplll). It uses a narrow C ABI shim over the
public fpLLL C++ API; Julia never depends directly on fpLLL's C++ ABI.

The package currently supports:

- LLL reduction of arbitrary-size integral row bases;
- all four fpLLL methods: wrapper, proved, heuristic, and fast;
- every public LLL parameter: `delta`, `eta`, floating type, MPFR precision,
  verbose output, early reduction, and the Siegel condition;
- optional unimodular transformation and inverse-transformation matrices;
- direct LLL reduction of integral Gram matrices;
- an independent, pure-Julia exact Gram--Schmidt checker;
- tests on random, triangular, q-ary, knapsack, integer-relation, and
  number-theoretic lattices through rank 32.

BKZ, HKZ, SVP, and the standalone fpLLL programs are intentionally outside the
current scope.

## Matrix convention

fpLLL regards the **rows** of a matrix as the lattice basis vectors. FPLLL.jl
uses the same mathematical convention:

```julia
B = BigInt[10 11; 11 12]
result = lll(B)
result.basis
```

Julia stores dense arrays in column-major memory, but this does not create a
transpose: the wrapper writes entries to the transfer buffer explicitly in
`(row, column)` order. Thus `result.basis[i, j]` corresponds exactly to fpLLL
row `i`, column `j`. Rectangular row bases are supported.

## Binary dependency and installation

This version uses `FPLLL_jll`; it no longer compiles a per-user shim against a
system fpLLL and has no `deps/build.jl`.  fpLLL and the C ABI shim are built
together by BinaryBuilder against JLL-provided GMP and MPFR.

For the pre-registration prototype, first build the supplied BinaryBuilder
recipe with `--deploy=local`, then `Pkg.develop` the generated `FPLLL_jll` into
this package.  The root-level `LOCAL_DEV.md` and `bootstrap_local.jl` automate
that workflow.  Once `FPLLL_jll` is registered, this becomes an ordinary Julia
dependency and users need no build step.

## GMP and memory ownership

Matrices no longer pass through a byte serialization format.  Input `BigInt`
objects are exposed only as read-only GMP-compatible `mpz` pointers while
protected from Julia's GC; the shim immediately copies them with its own GMP
into fpLLL-owned integers.  Results take the reverse route: while an opaque
fpLLL result handle is alive, Julia copies each read-only result `mpz` into a
fresh Julia-owned `BigInt` using Julia's own GMP.

Thus each direction performs one GMP integer copy, but no GMP allocation is
ever transferred between allocators.  A runtime check verifies compatible limb
width and rejects GMP builds using nail bits.

## Basis reduction

```julia
using FPLLL

B = BigInt[
    105 821 404;
    281 172 907;
    341 293 716;
]

result = lll(B;
    delta=0.99,
    eta=0.51,
    method=:wrapper,
    transform=true,
    inverse_transform=true,
)

R = result.basis
U = result.transform
Uinv = result.inverse_transform

@assert R == U * B
@assert Uinv * U == [1 0 0; 0 1 0; 0 0 1]
```

The input matrix is never modified, and all outputs use `BigInt`. Ordinary
Julia integer matrices are accepted and promoted. Values are copied directly through GMP's `mpz` ABI.  The pointers are
read-only and short-lived; each side owns the destination integer that it
allocates and clears.  There is no string or byte-stream serialization.

The same operation can be parameterized by an object:

```julia
parameters = LLLParams(
    delta=0.999,
    eta=0.501,
    method=:proved,
    float_type=:mpfr,
    precision=256,
    verbose=false,
    early_reduction=false,
    siegel=false,
)

result = lll(B, parameters; transform=true)
```

### Parameters

`method` accepts:

- `:wrapper` (fpLLL's adaptive, guaranteed wrapper);
- `:proved`;
- `:heuristic`;
- `:fast`.

`float_type` accepts:

- `:default`;
- `:double`;
- `:long_double` (alias `:longdouble`);
- `:dpe`;
- `:double_double` (alias `:dd`);
- `:quad_double` (alias `:qd`);
- `:mpfr`.

The package validates fpLLL's restrictions before crossing the C boundary. In
particular, wrapper mode uses `float_type=:default` and no explicit precision;
proved mode cannot be combined with early reduction; and explicit precision is
for MPFR. Optional backends depend on how fpLLL was compiled:

```julia
capabilities()
# (long_double = ..., dpe = ..., qd = ..., mpfr = true)
```

For wrapper mode, and for proved mode when fpLLL chooses its own precision,
fpLLL documents the rigorous final guarantee as `delta' = 2delta - 1` and
`eta' = 2eta - 1/2`. The helper `guaranteed_parameters(result.params)` returns
that pair, or `nothing` when the selected options alone do not establish a
proof-level guarantee. Passing an `LLLResult` also checks an explicit precision
against fpLLL's dimension-dependent bound. The bound itself is available as:

```julia
minimum_proved_precision(32; delta=0.99, eta=0.51)
```

## Direct Gram-matrix reduction

For an integral row basis `B`, its Gram matrix is
`G = B * transpose(B)`. It can be formed without overflow in a machine integer
type by using:

```julia
G = gram_matrix(B)
```

Reduce it directly with fpLLL's `MatGSOGram` path:

```julia
gram_result = lll_gram(G;
    delta=0.99,
    eta=0.51,
    float_type=:mpfr,
    precision=256,
    transform=true,
    inverse_transform=true,
)

Gred = gram_result.gram
U = gram_result.transform
@assert Gred == U * G * transpose(U)
```

This returns a reduced Gram matrix and a change-of-basis matrix. A Gram matrix
alone contains no coordinates for the vectors, so this operation cannot return
a coordinate basis unless one is separately supplied. The input must be an
integral symmetric positive-definite matrix. FPLLL.jl checks positive
definiteness exactly before entering fpLLL. Early reduction is
unavailable in this exact-Gram fpLLL path; the Siegel flag is supported.

## Independent verification

The test suite does not rely only on fpLLL's return status. These functions
compute exact rational Gram--Schmidt data in Julia and then check the relevant
inequalities at high precision:

```julia
is_lll_reduced(result.basis; delta=0.98, eta=0.52)
is_lll_reduced_gram(Gred; delta=0.99, eta=0.51)
```

Use `siegel=true` when checking output produced with the Siegel condition.

## Tests

Run:

```julia
import Pkg
Pkg.test("FPLLL")
```

The tests verify transformation identities and LLL inequalities for:

- asymmetric rectangular matrices, to catch orientation errors;
- integers far beyond 64 bits;
- wrapper, proved, heuristic, and fast methods;
- MPFR precision and all optional compiled floating backends;
- early reduction and Siegel flags;
- random full-rank and lower-triangular lattices;
- q-ary lattices;
- subset-sum/knapsack embeddings;
- integer-relation embeddings, including a forced exact relation;
- a polynomial relation at `sqrt(2)`;
- direct Gram reduction in ranks 2, 4, 8, 16, 24, and 32.

## Threading

fpLLL documents reductions on distinct objects as safe to run in parallel,
although concurrent work on the same fpLLL object is unsupported. Every call
through FPLLL.jl constructs fresh fpLLL matrices and reduction objects; the
Julia package itself does not add multithreading.

## License

The Julia package and shim are MIT licensed. fpLLL is a separate library under
LGPL-2.1-or-later; installing or distributing fpLLL remains subject to its own
license.
