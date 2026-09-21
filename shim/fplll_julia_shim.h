#ifndef FPLLL_JULIA_SHIM_H
#define FPLLL_JULIA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(FPLLL_JULIA_SHIM_BUILD)
#    define FPLLL_JULIA_API __declspec(dllexport)
#  else
#    define FPLLL_JULIA_API __declspec(dllimport)
#  endif
#else
#  define FPLLL_JULIA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum
{
  FPLLL_JULIA_CAP_LONG_DOUBLE = 1u << 0,
  FPLLL_JULIA_CAP_DPE         = 1u << 1,
  FPLLL_JULIA_CAP_QD          = 1u << 2,
  FPLLL_JULIA_CAP_MPFR        = 1u << 3
};

enum
{
  FPLLL_JULIA_RESULT_PRIMARY   = 0,
  FPLLL_JULIA_RESULT_TRANSFORM = 1,
  FPLLL_JULIA_RESULT_INVERSE   = 2
};

/* Opaque owner of fpLLL/GMP-side result matrices. */
typedef struct fplll_julia_result fplll_julia_result;

FPLLL_JULIA_API uint32_t fplll_julia_capabilities(void);
FPLLL_JULIA_API uint32_t fplll_julia_gmp_limb_bits(void);
FPLLL_JULIA_API uint32_t fplll_julia_gmp_nail_bits(void);

FPLLL_JULIA_API int fplll_julia_min_precision(
    int dimension,
    double delta,
    double eta);

/*
 * entries is a row-major array of read-only pointers to GMP-compatible mpz
 * structs.  The shim copies every source immediately into fpLLL-owned mpz_t
 * objects.  It never mutates a source mpz and never retains a source pointer
 * after this call returns.
 */
FPLLL_JULIA_API int fplll_julia_lll_mpz(
    const void *const *entries,
    size_t entry_count,
    int rows,
    int cols,
    double delta,
    double eta,
    int method,
    int float_type,
    int precision,
    int flags,
    int want_transform,
    int want_inverse_transform,
    fplll_julia_result **result_out,
    char **error_out);

FPLLL_JULIA_API int fplll_julia_lll_gram_mpz(
    const void *const *entries,
    size_t entry_count,
    int rows,
    int cols,
    double delta,
    double eta,
    int float_type,
    int precision,
    int flags,
    int want_transform,
    int want_inverse_transform,
    fplll_julia_result **result_out,
    char **error_out);

/* Query a matrix owned by result. Returns 0 on success. */
FPLLL_JULIA_API int fplll_julia_result_shape(
    const fplll_julia_result *result,
    int which,
    int *rows_out,
    int *cols_out);

/*
 * Fill entries_out, in row-major order, with read-only pointers to mpz values
 * owned by result.  These pointers are valid only until
 * fplll_julia_result_free(result).  Returns 0 on success.
 */
FPLLL_JULIA_API int fplll_julia_result_entry_ptrs(
    const fplll_julia_result *result,
    int which,
    const void **entries_out,
    size_t capacity);

FPLLL_JULIA_API void fplll_julia_result_free(fplll_julia_result *result);
FPLLL_JULIA_API void fplll_julia_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
