using Test
using Random
using LinearAlgebra
using FPLLL

include("test_utils.jl")

const RNG = MersenneTwister(0xF1_11_2026)

@testset "FPLLL.jl" begin
    @testset "capabilities and validation" begin
        caps = capabilities()
        @test caps.mpfr
        @test minimum_proved_precision(10) > 0
        @test_throws ArgumentError minimum_proved_precision(0)
        @test_throws ArgumentError LLLParams(delta=0.25)
        @test_throws ArgumentError LLLParams(delta=1.0)
        @test_throws ArgumentError LLLParams(eta=0.49)
        @test_throws ArgumentError LLLParams(method=:unknown)
        @test_throws ArgumentError LLLParams(method=:wrapper, float_type=:double)
        @test_throws ArgumentError LLLParams(method=:wrapper, precision=128)
        @test_throws ArgumentError LLLParams(method=:proved, early_reduction=true)
        @test_throws ArgumentError LLLParams(method=:fast, float_type=:mpfr)
        @test_throws ArgumentError lll(zeros(Int, 0, 2))
        @test_throws ArgumentError lll_gram(BigInt[1 2; 3 4])
        @test_throws ArgumentError lll_gram(BigInt[1 2; 2 1])
        @test_throws ArgumentError lll_gram(BigInt[1 0; 0 0])
    end

    @testset "direct GMP matrix transfer" begin
        # A one-row lattice cannot perform a nontrivial row operation, so this
        # is an exact end-to-end check of Julia -> shim -> fpLLL -> shim ->
        # Julia transfer, including zero, signs, limb boundaries, and integers
        # far larger than a machine word.
        one_row = reshape(BigInt[
            0,
            1,
            -1,
            127,
            128,
            255,
            256,
            -256,
            big(2)^64 - 1,
            -(big(2)^64),
            big(2)^521 + 17,
            -(big(3)^700 + 29),
            big(2)^4096 + 123,
            -(big(5)^1800 + 7),
        ], 1, 14)
        roundtrip = lll(one_row)
        @test roundtrip.basis == one_row

        # Non-square input catches a row/column-order mistake in the pointer
        # vector independently of the exact one-row round trip above.
        rectangular = BigInt[100 1 0; 1 0 1]
        rectangular_result = lll(rectangular;
                                 transform=true,
                                 inverse_transform=true)
        check_basis_result(rectangular, rectangular_result; check_inverse=true)
    end

    @testset "orientation, transforms, and arbitrary precision" begin
        basis = BigInt[10 11; 11 12]
        saved = copy(basis)
        result = lll(basis; transform=true, inverse_transform=true)
        @test basis == saved
        check_basis_result(basis, result; check_inverse=true)

        plain = lll(basis)
        @test plain.transform === nothing
        @test plain.inverse_transform === nothing
        @test is_lll_reduced(plain.basis; delta=0.98, eta=0.52)

        inverse_only = lll(basis; inverse_transform=true)
        @test inverse_only.transform === nothing
        @test inverse_only.inverse_transform !== nothing
        @test inverse_only.inverse_transform * inverse_only.basis == basis

        # Recheck the non-square transform path alongside ordinary reductions.
        rectangular = BigInt[100 1 0; 1 0 1]
        rectangular_result = lll(rectangular; transform=true, inverse_transform=true)
        check_basis_result(rectangular, rectangular_result; check_inverse=true)

        huge = BigInt[
            big(2)^300 + 17  big(3)^180 - 5  1;
            big(2)^299 - 11  big(3)^180 + 8  2;
            7                13                big(5)^130;
        ]
        huge_result = lll(huge; transform=true, inverse_transform=true)
        check_basis_result(huge, huge_result; check_inverse=true)
    end

    @testset "methods, floating types, and flags" begin
        basis = random_full_rank_lattice(RNG, 10; bits=24)

        wrapper = lll(basis; method=:wrapper, transform=true)
        check_basis_result(basis, wrapper)

        proved_precision = minimum_proved_precision(size(basis, 1))
        proved = lll(basis; method=:proved, float_type=:mpfr,
                     precision=proved_precision, transform=true)
        check_basis_result(basis, proved)

        heuristic = lll(basis; method=:heuristic, float_type=:double,
                        transform=true)
        check_basis_result(basis, heuristic)

        fast = lll(basis; method=:fast, float_type=:double,
                   transform=true)
        check_basis_result(basis, fast)

        early = lll(basis; method=:wrapper, early_reduction=true,
                    transform=true)
        check_basis_result(basis, early)

        siegel = lll(basis; method=:wrapper, siegel=true, transform=true)
        check_basis_result(basis, siegel)

        @test LLLParams(method=:heuristic, float_type=:longdouble).float_type === :long_double
        @test LLLParams(method=:fast, float_type=:dd).float_type === :double_double
        @test LLLParams(method=:fast, float_type=:qd).float_type === :quad_double

        caps = capabilities()
        if caps.long_double
            result = lll(basis; method=:heuristic,
                         float_type=:long_double, transform=true)
            check_basis_result(basis, result)
        end
        if caps.dpe
            result = lll(basis; method=:heuristic,
                         float_type=:dpe, transform=true)
            check_basis_result(basis, result)
        end
        if caps.qd
            dd = lll(basis; method=:fast,
                     float_type=:double_double, transform=true)
            check_basis_result(basis, dd)
            qd = lll(basis; method=:fast,
                     float_type=:quad_double, transform=true)
            check_basis_result(basis, qd)
        end
    end

    @testset "random and structured lattices through dimension 32" begin
        for n in (2, 4, 8, 16, 24, 32)
            basis = random_full_rank_lattice(RNG, n; bits=18 + n ÷ 4)
            result = lll(basis; transform=true)
            check_basis_result(basis, result)
        end

        for n in (4, 8, 16, 32)
            basis = triangular_lattice(n)
            result = lll(basis; transform=true)
            check_basis_result(basis, result)
        end

        for n in (4, 8, 16, 31)
            basis = qary_lattice(RNG, n)
            result = lll(basis; transform=true)
            check_basis_result(basis, result)
        end
    end

    @testset "knapsack and relation-finding lattices" begin
        for number_of_items in (3, 7, 15, 31)
            basis = knapsack_lattice(RNG, number_of_items)
            @test size(basis, 1) <= 32
            result = lll(basis; transform=true)
            check_basis_result(basis, result)
        end

        for n in (3, 8, 16, 32)
            values = [BigInt(rand(RNG, 10^5:10^7)) for _ in 1:n]
            # Force a genuine relation among the first three values.
            n >= 3 && (values[3] = values[1] + values[2])
            basis = relation_lattice(values; scale=big(10)^8)
            result = lll(basis; transform=true)
            check_basis_result(basis, result)
            exact_relations = [row for row in eachrow(result.basis)
                               if iszero(row[end])]
            @test !isempty(exact_relations)
            @test all(sum(row[i] * values[i] for i in eachindex(values)) == 0
                      for row in exact_relations)
            n == 3 && @test any(sum(abs2, row[1:end-1]) <= 3
                                for row in exact_relations)
        end

        # Number-theoretic relation: x^2 - 2 = 0 at x = sqrt(2).
        scale = big(10)^35
        approximation = isqrt(2 * scale^2)
        sqrt2 = BigInt[
            1 0 0 scale;
            0 1 0 approximation;
            0 0 1 2scale;
        ]
        result = lll(sqrt2; transform=true)
        check_basis_result(sqrt2, result)
        @test any(iszero(row[4]) && sum(abs2, row[1:3]) <= 5
                  for row in eachrow(result.basis))
    end

    @testset "direct Gram-matrix reduction" begin
        for n in (2, 4, 8, 16, 24, 32)
            basis = random_full_rank_lattice(RNG, n; bits=14 + n ÷ 5)
            gram = gram_matrix(basis)
            saved = copy(gram)
            result = lll_gram(gram; float_type=:mpfr, precision=256,
                              transform=true,
                              inverse_transform=(n <= 8))
            @test gram == saved
            check_gram_result(gram, result; check_inverse=(n <= 8))
        end

        basis = triangular_lattice(12)
        gram = gram_matrix(basis)
        result = lll_gram(gram; float_type=:double,
                          transform=true, siegel=true)
        check_gram_result(gram, result)

        small_gram = gram_matrix(BigInt[10 11; 11 12])
        inverse_only = lll_gram(small_gram; inverse_transform=true,
                                float_type=:mpfr, precision=128)
        @test inverse_only.transform === nothing
        @test inverse_only.inverse_transform !== nothing
        @test inverse_only.inverse_transform * inverse_only.gram *
              transpose(inverse_only.inverse_transform) == small_gram

        @test !is_lll_reduced_gram(BigInt[1 2; 2 1])
    end
end
