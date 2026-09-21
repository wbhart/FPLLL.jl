function identity_bigint(n::Int)
    result = zeros(BigInt, n, n)
    for i in 1:n
        result[i, i] = 1
    end
    return result
end

function random_unimodular(rng::AbstractRNG, n::Int; steps::Int=8n, coefficient::Int=4)
    matrix = identity_bigint(n)
    for _ in 1:steps
        operation = rand(rng, 1:3)
        i, j = rand(rng, 1:n, 2)
        i == j && continue
        if operation == 1
            temporary = copy(matrix[i, :])
            matrix[i, :] = matrix[j, :]
            matrix[j, :] = temporary
        elseif operation == 2
            matrix[i, :] .*= -1
        else
            multiplier = rand(rng, -coefficient:coefficient)
            iszero(multiplier) && continue
            matrix[i, :] .+= multiplier .* matrix[j, :]
        end
    end
    return matrix
end

function random_full_rank_lattice(rng::AbstractRNG, n::Int; bits::Int=18)
    diagonal = Matrix{BigInt}(undef, n, n)
    fill!(diagonal, 0)
    scale = big(1) << bits
    for i in 1:n
        diagonal[i, i] = scale + rand(rng, 1:10_000)
    end
    for i in 2:n, j in 1:i-1
        diagonal[i, j] = rand(rng, -10_000:10_000)
    end
    return random_unimodular(rng, n; steps=12n) * diagonal
end

function triangular_lattice(n::Int)
    basis = zeros(BigInt, n, n)
    for i in 1:n
        basis[i, i] = big(2)^(n - i + 5)
        for j in 1:i-1
            basis[i, j] = (-1)^(i + j) * BigInt(i * j + 1)
        end
    end
    return basis
end

# A standard subset-sum/knapsack embedding. The first n rows encode choices;
# the final row supplies a target offset. It has full row rank.
function knapsack_lattice(rng::AbstractRNG, n::Int)
    weights = [BigInt(rand(rng, 1:10^7)) for _ in 1:n]
    target = sum(weights[i] for i in 1:2:n)
    scale = big(2)^(max(12, n ÷ 2))
    basis = zeros(BigInt, n + 1, n + 2)
    for i in 1:n
        basis[i, i] = 2
        basis[i, n + 1] = 2 * scale * weights[i]
        basis[i, n + 2] = 1
    end
    basis[n + 1, 1:n] .= 1
    basis[n + 1, n + 1] = 2 * scale * target
    basis[n + 1, n + 2] = 1
    return basis
end

# Integer-relation embedding for integers a_i. A short vector corresponds to
# coefficients c_i for which sum(c_i*a_i) is small or zero.
function relation_lattice(values::Vector{BigInt}; scale::BigInt=big(10)^12)
    n = length(values)
    basis = zeros(BigInt, n, n + 1)
    for i in 1:n
        basis[i, i] = 1
        basis[i, n + 1] = scale * values[i]
    end
    return basis
end

# A q-ary lattice in systematic form [qI 0; A I].
function qary_lattice(rng::AbstractRNG, n::Int; q::Int=65537)
    left = n ÷ 2
    right = n - left
    basis = zeros(BigInt, n, n)
    for i in 1:left
        basis[i, i] = q
    end
    for i in 1:right
        for j in 1:left
            basis[left + i, j] = rand(rng, 0:q-1)
        end
        basis[left + i, left + i] = 1
    end
    return basis
end

function check_basis_result(original, result; check_inverse::Bool=false)
    rows = size(original, 1)
    @test result.transform !== nothing
    @test result.basis == result.transform * Matrix{BigInt}(original)
    if check_inverse
        @test result.inverse_transform !== nothing
        @test result.inverse_transform * result.transform == identity_bigint(rows)
        @test result.transform * result.inverse_transform == identity_bigint(rows)
    end

    guarantee = guaranteed_parameters(result)
    checked = isnothing(guarantee) ? (delta=result.params.delta,
                                        eta=result.params.eta) : guarantee
    @test is_lll_reduced(result.basis;
                         delta=checked.delta,
                         eta=checked.eta,
                         siegel=result.params.siegel,
                         precision=384)
    return nothing
end

function check_gram_result(original, result; check_inverse::Bool=false)
    rows = size(original, 1)
    @test result.transform !== nothing
    @test result.gram == result.transform * Matrix{BigInt}(original) * transpose(result.transform)
    if check_inverse
        @test result.inverse_transform !== nothing
        @test result.inverse_transform * result.transform == identity_bigint(rows)
        @test result.transform * result.inverse_transform == identity_bigint(rows)
    end
    @test is_lll_reduced_gram(result.gram;
                              delta=result.delta,
                              eta=result.eta,
                              siegel=result.siegel,
                              precision=384)
    return nothing
end
