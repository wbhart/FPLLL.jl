module FPLLL

using FPLLL_jll

export FPLLLException,
       LLLParams,
       LLLResult,
       GramLLLResult,
       capabilities,
       guaranteed_parameters,
       gram_matrix,
       is_lll_reduced,
       is_lll_reduced_gram,
       minimum_proved_precision,
       lll,
       lll_gram

const libfplll_julia_shim = FPLLL_jll.libfplll_julia_shim

"""An error reported by fpLLL or by the C ABI shim."""
struct FPLLLException <: Exception
    message::String
    status::Int
end

function Base.showerror(io::IO, error::FPLLLException)
    print(io, error.message)
    error.status == 0 || print(io, " (status ", error.status, ")")
end

"""
    LLLParams(; delta=0.99, eta=0.51, method=:wrapper,
                float_type=:default, precision=0,
                verbose=false, early_reduction=false, siegel=false)

Parameters accepted by fpLLL's LLL wrapper.

`method` is one of `:wrapper`, `:proved`, `:heuristic`, or `:fast`.
`float_type` is one of `:default`, `:double`, `:long_double`, `:dpe`,
`:double_double`, `:quad_double`, or `:mpfr`.
"""
struct LLLParams
    delta::Float64
    eta::Float64
    method::Symbol
    float_type::Symbol
    precision::Int
    verbose::Bool
    early_reduction::Bool
    siegel::Bool
end

function LLLParams(; delta::Real=0.99,
                     eta::Real=0.51,
                     method::Symbol=:wrapper,
                     float_type::Symbol=:default,
                     precision::Integer=0,
                     verbose::Bool=false,
                     early_reduction::Bool=false,
                     siegel::Bool=false)
    params = LLLParams(Float64(delta), Float64(eta), method,
                       _canonical_float_type(float_type), Int(precision),
                       verbose, early_reduction, siegel)
    _validate_params(params)
    return params
end

"""The result of reducing a row-basis matrix."""
struct LLLResult
    basis::Matrix{BigInt}
    transform::Union{Nothing, Matrix{BigInt}}
    inverse_transform::Union{Nothing, Matrix{BigInt}}
    params::LLLParams
end

"""The result of reducing an integral Gram matrix."""
struct GramLLLResult
    gram::Matrix{BigInt}
    transform::Union{Nothing, Matrix{BigInt}}
    inverse_transform::Union{Nothing, Matrix{BigInt}}
    delta::Float64
    eta::Float64
    float_type::Symbol
    precision::Int
    verbose::Bool
    siegel::Bool
end

const _METHODS = Dict{Symbol, Cint}(
    :wrapper => 0,
    :proved => 1,
    :heuristic => 2,
    :fast => 3,
)

const _FLOAT_TYPES = Dict{Symbol, Cint}(
    :default => 0,
    :double => 1,
    :long_double => 2,
    :dpe => 3,
    :double_double => 4,
    :quad_double => 5,
    :mpfr => 6,
)

const _FLAG_VERBOSE = Cint(1)
const _FLAG_EARLY_REDUCTION = Cint(2)
const _FLAG_SIEGEL = Cint(4)

function _canonical_float_type(value::Symbol)
    value === :longdouble && return :long_double
    value === :dd && return :double_double
    value === :qd && return :quad_double
    return value
end

const _gmp_abi_checked = Ref(false)

function _require_library()
    _gmp_abi_checked[] && return nothing

    shim_limb_bits = ccall(
        (:fplll_julia_gmp_limb_bits, libfplll_julia_shim), UInt32, ())
    shim_nail_bits = ccall(
        (:fplll_julia_gmp_nail_bits, libfplll_julia_shim), UInt32, ())
    julia_limb_bits = UInt32(Base.GMP.BITS_PER_LIMB)

    shim_limb_bits == julia_limb_bits || throw(FPLLLException(
        "GMP ABI mismatch: fpLLL uses $(shim_limb_bits)-bit limbs but Julia uses $(julia_limb_bits)-bit limbs.",
        0,
    ))
    iszero(shim_nail_bits) || throw(FPLLLException(
        "Unsupported GMP ABI: fpLLL's GMP was built with $(shim_nail_bits) nail bits per limb.",
        0,
    ))

    _gmp_abi_checked[] = true
    return nothing
end

function __init__()
    _gmp_abi_checked[] = false
    return nothing
end

"""
    capabilities()

Return the optional floating-point backends enabled in the linked fpLLL build.
"""
function capabilities()
    _require_library()
    bits = ccall((:fplll_julia_capabilities, libfplll_julia_shim), UInt32, ())
    return (
        long_double = !iszero(bits & UInt32(1 << 0)),
        dpe = !iszero(bits & UInt32(1 << 1)),
        qd = !iszero(bits & UInt32(1 << 2)),
        mpfr = !iszero(bits & UInt32(1 << 3)),
    )
end

"""
    minimum_proved_precision(dimension; delta=0.99, eta=0.51) -> Int

Return fpLLL's dimension-dependent minimum floating-point precision for a
proved LLL reduction with the given parameters.
"""
function minimum_proved_precision(dimension::Integer;
                                  delta::Real=0.99,
                                  eta::Real=0.51)
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    dimension <= typemax(Cint) || throw(ArgumentError("dimension is too large for fpLLL"))
    delta_float = Float64(delta)
    eta_float = Float64(eta)
    _validate_common(delta_float, eta_float, :mpfr, 0)
    _require_library()
    precision = ccall(
        (:fplll_julia_min_precision, libfplll_julia_shim),
        Cint,
        (Cint, Cdouble, Cdouble),
        Cint(dimension), delta_float, eta_float,
    )
    precision > 0 || throw(FPLLLException(
        "fpLLL could not determine a minimum proved-mode precision.", Int(precision)))
    return Int(precision)
end

function _validate_common(delta::Float64, eta::Float64,
                          float_type::Symbol, precision::Int)
    isfinite(delta) || throw(ArgumentError("delta must be finite"))
    isfinite(eta) || throw(ArgumentError("eta must be finite"))
    0.25 < delta < 1.0 || throw(ArgumentError("delta must satisfy 0.25 < delta < 1"))
    0.5 <= eta < sqrt(delta) ||
        throw(ArgumentError("eta must satisfy 0.5 <= eta < sqrt(delta)"))
    haskey(_FLOAT_TYPES, float_type) ||
        throw(ArgumentError("unknown fpLLL floating-point type: $float_type"))
    precision >= 0 || throw(ArgumentError("precision must be nonnegative"))
    precision <= typemax(Cint) || throw(ArgumentError("precision is too large for fpLLL"))
    if precision != 0 && !(float_type in (:default, :mpfr))
        throw(ArgumentError("an explicit precision requires float_type=:default or :mpfr"))
    end
    return nothing
end

function _validate_params(params::LLLParams)
    _validate_common(params.delta, params.eta,
                     params.float_type, params.precision)
    haskey(_METHODS, params.method) ||
        throw(ArgumentError("unknown fpLLL LLL method: $(params.method)"))

    if params.method === :wrapper
        params.float_type === :default ||
            throw(ArgumentError("method=:wrapper requires float_type=:default"))
        params.precision == 0 ||
            throw(ArgumentError("method=:wrapper does not accept an explicit precision"))
    end
    if params.method === :proved && params.early_reduction
        throw(ArgumentError("fpLLL does not implement proved LLL with early reduction"))
    end
    if params.method === :fast &&
       !(params.float_type in (:default, :double, :long_double,
                               :double_double, :quad_double))
        throw(ArgumentError("method=:fast requires a fixed-precision floating-point type"))
    end
    return nothing
end

function _flags(; verbose::Bool=false,
                  early_reduction::Bool=false,
                  siegel::Bool=false)
    flags = Cint(0)
    verbose && (flags |= _FLAG_VERBOSE)
    early_reduction && (flags |= _FLAG_EARLY_REDUCTION)
    siegel && (flags |= _FLAG_SIEGEL)
    return flags
end

function _big_matrix(matrix::AbstractMatrix{<:Integer})
    rows, cols = size(matrix)
    rows > 0 || throw(ArgumentError("a lattice matrix must have at least one row"))
    cols > 0 || throw(ArgumentError("a lattice matrix must have at least one column"))
    rows <= typemax(Cint) || throw(ArgumentError("the row count exceeds fpLLL's limit"))
    cols <= typemax(Cint) || throw(ArgumentError("the column count exceeds fpLLL's limit"))
    result = Matrix{BigInt}(undef, rows, cols)
    for (output_i, input_i) in enumerate(axes(matrix, 1)),
        (output_j, input_j) in enumerate(axes(matrix, 2))
        result[output_i, output_j] = BigInt(matrix[input_i, input_j])
    end
    return result
end

# Fast GMP bridge.
#
# Input BigInts are exposed to the C shim only as read-only mpz-compatible
# pointers.  The shim copies them immediately into fpLLL-owned integers.  For
# outputs the shim exposes read-only pointers to fpLLL-owned mpz values while
# an opaque result handle is alive; Julia copies those values with Julia's own
# GMP before the handle is freed.  No GMP allocation changes allocator owner.
const _RESULT_PRIMARY = Cint(0)
const _RESULT_TRANSFORM = Cint(1)
const _RESULT_INVERSE = Cint(2)

function _matrix_mpz_pointers(matrix::Matrix{BigInt})
    rows, cols = size(matrix)
    pointers = Vector{Ptr{Cvoid}}(undef, Base.checked_mul(rows, cols))
    k = 1
    for i in 1:rows, j in 1:cols
        pointers[k] = Ptr{Cvoid}(pointer_from_objref(matrix[i, j]))
        k += 1
    end
    return pointers
end

function _copy_mpz(source::Ptr{Cvoid})
    source == C_NULL && throw(FPLLLException(
        "The fpLLL shim returned a null mpz pointer.", 0))
    value = BigInt()
    ccall(
        (:__gmpz_set, Base.GMP.libgmp),
        Cvoid,
        (Ref{BigInt}, Ptr{Cvoid}),
        value, source,
    )
    return value
end

function _copy_result_matrix(handle::Ptr{Cvoid}, which::Cint;
                             expected_rows::Union{Nothing, Int}=nothing,
                             expected_cols::Union{Nothing, Int}=nothing)
    handle == C_NULL && throw(FPLLLException(
        "The fpLLL shim returned a null result handle.", 0))

    rows_ref = Ref{Cint}(0)
    cols_ref = Ref{Cint}(0)
    status = ccall(
        (:fplll_julia_result_shape, libfplll_julia_shim),
        Cint,
        (Ptr{Cvoid}, Cint, Ref{Cint}, Ref{Cint}),
        handle, which, rows_ref, cols_ref,
    )
    status == 0 || throw(FPLLLException(
        "The fpLLL shim could not expose a requested result matrix.", Int(status)))

    rows = Int(rows_ref[])
    cols = Int(cols_ref[])
    rows > 0 && cols > 0 || throw(FPLLLException(
        "The fpLLL shim returned invalid matrix dimensions.", 0))
    expected_rows === nothing || rows == expected_rows || throw(FPLLLException(
        "The fpLLL shim returned $rows rows; expected $expected_rows.", 0))
    expected_cols === nothing || cols == expected_cols || throw(FPLLLException(
        "The fpLLL shim returned $cols columns; expected $expected_cols.", 0))

    count = Base.checked_mul(rows, cols)
    pointers = Vector{Ptr{Cvoid}}(undef, count)
    status = ccall(
        (:fplll_julia_result_entry_ptrs, libfplll_julia_shim),
        Cint,
        (Ptr{Cvoid}, Cint, Ptr{Ptr{Cvoid}}, Csize_t),
        handle, which, pointers, Csize_t(count),
    )
    status == 0 || throw(FPLLLException(
        "The fpLLL shim could not expose result matrix entries.", Int(status)))

    result = Matrix{BigInt}(undef, rows, cols)
    k = 1
    for i in 1:rows, j in 1:cols
        result[i, j] = _copy_mpz(pointers[k])
        k += 1
    end
    return result
end

function _take_string!(reference::Ref{Ptr{Cchar}})
    pointer = reference[]
    pointer == C_NULL && return nothing
    try
        return unsafe_string(pointer)
    finally
        ccall((:fplll_julia_free, libfplll_julia_shim), Cvoid, (Ptr{Cvoid},), pointer)
        reference[] = C_NULL
    end
end

function _free_pointer!(reference::Ref{Ptr{Cchar}})
    pointer = reference[]
    if pointer != C_NULL
        ccall((:fplll_julia_free, libfplll_julia_shim), Cvoid, (Ptr{Cvoid},), pointer)
        reference[] = C_NULL
    end
    return nothing
end

function _free_result!(reference::Ref{Ptr{Cvoid}})
    pointer = reference[]
    if pointer != C_NULL
        ccall((:fplll_julia_result_free, libfplll_julia_shim),
              Cvoid, (Ptr{Cvoid},), pointer)
        reference[] = C_NULL
    end
    return nothing
end

"""
    lll(basis; kwargs...) -> LLLResult
    lll(basis, params::LLLParams; transform=false, inverse_transform=false)

LLL-reduce a matrix whose **rows** are lattice basis vectors. The input is not
modified. Set `transform=true` to obtain `U` with `result.basis == U * basis`,
and `inverse_transform=true` to obtain `U⁻¹` as well.
"""
function lll(basis::AbstractMatrix{<:Integer};
             delta::Real=0.99,
             eta::Real=0.51,
             method::Symbol=:wrapper,
             float_type::Symbol=:default,
             precision::Integer=0,
             verbose::Bool=false,
             early_reduction::Bool=false,
             siegel::Bool=false,
             transform::Bool=false,
             inverse_transform::Bool=false)
    params = LLLParams(; delta, eta, method, float_type, precision,
                       verbose, early_reduction, siegel)
    return lll(basis, params; transform, inverse_transform)
end

function lll(basis::AbstractMatrix{<:Integer}, params::LLLParams;
             transform::Bool=false,
             inverse_transform::Bool=false)
    _require_library()
    _validate_params(params)

    input = _big_matrix(basis)
    rows, cols = size(input)
    pointers = _matrix_mpz_pointers(input)
    result_handle = Ref{Ptr{Cvoid}}(C_NULL)
    error_pointer = Ref{Ptr{Cchar}}(C_NULL)

    status = GC.@preserve input pointers begin
        ccall(
            (:fplll_julia_lll_mpz, libfplll_julia_shim),
            Cint,
            (Ptr{Ptr{Cvoid}}, Csize_t, Cint, Cint,
             Cdouble, Cdouble, Cint, Cint, Cint, Cint,
             Cint, Cint, Ref{Ptr{Cvoid}}, Ref{Ptr{Cchar}}),
            pointers, Csize_t(length(pointers)), Cint(rows), Cint(cols),
            params.delta, params.eta,
            _METHODS[params.method], _FLOAT_TYPES[params.float_type],
            Cint(params.precision),
            _flags(; verbose=params.verbose,
                     early_reduction=params.early_reduction,
                     siegel=params.siegel),
            Cint(transform), Cint(inverse_transform),
            result_handle, error_pointer,
        )
    end

    try
        if status != 0
            message = _take_string!(error_pointer)
            throw(FPLLLException(something(message, "fpLLL reduction failed"), Int(status)))
        end

        reduced = _copy_result_matrix(
            result_handle[], _RESULT_PRIMARY;
            expected_rows=rows, expected_cols=cols)

        computed_transform = transform ? _copy_result_matrix(
            result_handle[], _RESULT_TRANSFORM;
            expected_rows=rows, expected_cols=rows) : nothing

        computed_inverse = inverse_transform ? _copy_result_matrix(
            result_handle[], _RESULT_INVERSE;
            expected_rows=rows, expected_cols=rows) : nothing

        return LLLResult(reduced, computed_transform, computed_inverse, params)
    finally
        _free_result!(result_handle)
        _free_pointer!(error_pointer)
    end
end

"""
    lll_gram(gram; delta=0.99, eta=0.51, float_type=:mpfr,
             precision=0, verbose=false, siegel=false,
             transform=false, inverse_transform=false) -> GramLLLResult

LLL-reduce an integral Gram matrix directly. If `G` is the input and `U` is
the returned transformation, the reduced Gram matrix is `U * G * transpose(U)`.
The input must be symmetric positive definite. Positive definiteness is checked
exactly before the matrix is passed to fpLLL.
"""
function lll_gram(gram::AbstractMatrix{<:Integer};
                  delta::Real=0.99,
                  eta::Real=0.51,
                  float_type::Symbol=:mpfr,
                  precision::Integer=0,
                  verbose::Bool=false,
                  siegel::Bool=false,
                  transform::Bool=false,
                  inverse_transform::Bool=false)
    canonical_float = _canonical_float_type(float_type)
    delta_float = Float64(delta)
    eta_float = Float64(eta)
    precision_int = Int(precision)
    _validate_common(delta_float, eta_float, canonical_float, precision_int)

    _require_library()
    input = _big_matrix(gram)
    rows, cols = size(input)
    rows == cols || throw(ArgumentError("a Gram matrix must be square"))
    for i in 1:rows, j in 1:i-1
        input[i, j] == input[j, i] ||
            throw(ArgumentError("a Gram matrix must be symmetric"))
    end
    _gram_schmidt_from_gram(input) === nothing &&
        throw(ArgumentError("a Gram matrix must be positive definite"))

    pointers = _matrix_mpz_pointers(input)
    result_handle = Ref{Ptr{Cvoid}}(C_NULL)
    error_pointer = Ref{Ptr{Cchar}}(C_NULL)

    status = GC.@preserve input pointers begin
        ccall(
            (:fplll_julia_lll_gram_mpz, libfplll_julia_shim),
            Cint,
            (Ptr{Ptr{Cvoid}}, Csize_t, Cint, Cint,
             Cdouble, Cdouble, Cint, Cint, Cint,
             Cint, Cint, Ref{Ptr{Cvoid}}, Ref{Ptr{Cchar}}),
            pointers, Csize_t(length(pointers)), Cint(rows), Cint(cols),
            delta_float, eta_float,
            _FLOAT_TYPES[canonical_float], Cint(precision_int),
            _flags(; verbose, siegel),
            Cint(transform), Cint(inverse_transform),
            result_handle, error_pointer,
        )
    end

    try
        if status != 0
            message = _take_string!(error_pointer)
            throw(FPLLLException(something(message, "fpLLL Gram reduction failed"), Int(status)))
        end

        reduced = _copy_result_matrix(
            result_handle[], _RESULT_PRIMARY;
            expected_rows=rows, expected_cols=rows)

        computed_transform = transform ? _copy_result_matrix(
            result_handle[], _RESULT_TRANSFORM;
            expected_rows=rows, expected_cols=rows) : nothing

        computed_inverse = inverse_transform ? _copy_result_matrix(
            result_handle[], _RESULT_INVERSE;
            expected_rows=rows, expected_cols=rows) : nothing

        return GramLLLResult(reduced, computed_transform, computed_inverse,
                             delta_float, eta_float, canonical_float,
                             precision_int, verbose, siegel)
    finally
        _free_result!(result_handle)
        _free_pointer!(error_pointer)
    end
end

"""
    guaranteed_parameters(params)

Return fpLLL's documented proof-level reduction parameters, or `nothing`
when the selected options do not by themselves provide such a guarantee.
Wrapper mode, and proved mode with fpLLL-selected precision, guarantee
`(2delta - 1, 2eta - 1/2)`. Pass a dimension as a second argument, or pass an
`LLLResult`, to check an explicit proved-mode precision against fpLLL's
minimum. Heuristic and fast modes are not proved.
"""
function guaranteed_parameters(params::LLLParams)
    if params.method === :wrapper ||
       (params.method === :proved && params.precision == 0)
        return (delta=2 * params.delta - 1,
                eta=2 * params.eta - 0.5)
    end
    return nothing
end

function guaranteed_parameters(params::LLLParams, dimension::Integer)
    if params.method === :wrapper
        return (delta=2 * params.delta - 1,
                eta=2 * params.eta - 0.5)
    elseif params.method === :proved
        enough_precision = params.precision == 0 ||
            params.precision >= minimum_proved_precision(
                dimension; delta=params.delta, eta=params.eta)
        if enough_precision
            return (delta=2 * params.delta - 1,
                    eta=2 * params.eta - 0.5)
        end
    end
    return nothing
end

guaranteed_parameters(result::LLLResult) =
    guaranteed_parameters(result.params, size(result.basis, 1))

"""Compute the integral row Gram matrix `basis * transpose(basis)`."""
function gram_matrix(basis::AbstractMatrix{<:Integer})
    input = _big_matrix(basis)
    rows, cols = size(input)
    gram = Matrix{BigInt}(undef, rows, rows)
    for i in 1:rows, j in 1:i
        value = BigInt(0)
        for k in 1:cols
            value += input[i, k] * input[j, k]
        end
        gram[i, j] = value
        gram[j, i] = value
    end
    return gram
end

function _rational(value::BigInt)
    return Rational{BigInt}(value, BigInt(1))
end

# Exact LDL^T/Gram--Schmidt data. This is deliberately independent of fpLLL,
# and is used both by users and by the test suite to validate reductions.
function _gram_schmidt_from_gram(gram::Matrix{BigInt})
    n = size(gram, 1)
    mu = fill(Rational{BigInt}(0), n, n)
    norms = fill(Rational{BigInt}(0), n)

    for i in 1:n
        for j in 1:i-1
            numerator = _rational(gram[i, j])
            for k in 1:j-1
                numerator -= mu[i, k] * mu[j, k] * norms[k]
            end
            iszero(norms[j]) && return nothing
            mu[i, j] = numerator / norms[j]
        end

        norm = _rational(gram[i, i])
        for k in 1:i-1
            norm -= mu[i, k]^2 * norms[k]
        end
        norm > 0 || return nothing
        norms[i] = norm
    end
    return mu, norms
end

function _bigfloat(value::Rational{BigInt})
    return BigFloat(numerator(value)) / BigFloat(denominator(value))
end

"""
    is_lll_reduced(basis; delta=0.99, eta=0.51, siegel=false,
                   precision=256) -> Bool

Independently check the size-reduction and Lovász (or Siegel) conditions using
exact rational Gram--Schmidt data and high-precision comparisons.
"""
function is_lll_reduced(basis::AbstractMatrix{<:Integer}; kwargs...)
    return is_lll_reduced_gram(gram_matrix(basis); kwargs...)
end

"""
    is_lll_reduced_gram(gram; delta=0.99, eta=0.51, siegel=false,
                        precision=256) -> Bool

Check LLL conditions directly from an integral Gram matrix.
"""
function is_lll_reduced_gram(gram::AbstractMatrix{<:Integer};
                             delta::Real=0.99,
                             eta::Real=0.51,
                             siegel::Bool=false,
                             precision::Integer=256)
    delta_float = Float64(delta)
    eta_float = Float64(eta)
    _validate_common(delta_float, eta_float, :mpfr, 0)
    precision >= 64 || throw(ArgumentError("verification precision must be at least 64 bits"))

    input = _big_matrix(gram)
    rows, cols = size(input)
    rows == cols || return false
    for i in 1:rows, j in 1:i-1
        input[i, j] == input[j, i] || return false
    end

    data = _gram_schmidt_from_gram(input)
    data === nothing && return false
    mu, norms = data

    return setprecision(BigFloat, Int(precision)) do
        eta_big = BigFloat(eta_float)
        delta_big = BigFloat(delta_float)
        tolerance = ldexp(BigFloat(1), -Int(precision) + 16)

        for i in 2:rows
            for j in 1:i-1
                coefficient = abs(_bigfloat(mu[i, j]))
                scale = max(coefficient, eta_big, BigFloat(1))
                coefficient <= eta_big + tolerance * scale || return false
            end

            lhs = _bigfloat(norms[i])
            if siegel
                rhs = (delta_big - eta_big^2) * _bigfloat(norms[i - 1])
            else
                previous_mu = _bigfloat(mu[i, i - 1])
                rhs = (delta_big - previous_mu^2) * _bigfloat(norms[i - 1])
            end
            scale = max(abs(lhs), abs(rhs), BigFloat(1))
            lhs + tolerance * scale >= rhs || return false
        end
        return true
    end
end

end # module
