#define FPLLL_JULIA_SHIM_BUILD
#include "fplll_julia_shim.h"

#include <fplll/fplll.h>
#include <gmp.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

using namespace fplll;

struct fplll_julia_result
{
  explicit fplll_julia_result(int rows, int cols)
      : primary(rows, cols), has_transform(false), has_inverse(false)
  {
  }

  ZZ_mat<mpz_t> primary;
  ZZ_mat<mpz_t> transform;
  ZZ_mat<mpz_t> inverse;
  bool has_transform;
  bool has_inverse;
};

namespace
{

char *copy_string(const std::string &value)
{
  char *result = static_cast<char *>(std::malloc(value.size() + 1));
  if (result == nullptr)
    throw std::bad_alloc();
  std::memcpy(result, value.c_str(), value.size() + 1);
  return result;
}

void set_error(char **error_out, const std::string &message) noexcept
{
  if (error_out == nullptr)
    return;
  try
  {
    *error_out = copy_string(message);
  }
  catch (...)
  {
    *error_out = nullptr;
  }
}

void clear_outputs(fplll_julia_result **result_out, char **error_out) noexcept
{
  if (result_out != nullptr)
    *result_out = nullptr;
  if (error_out != nullptr)
    *error_out = nullptr;
}

std::string status_message(int status)
{
  if (status >= 0 && status < RED_STATUS_MAX)
    return std::string("fpLLL reduction failed: ") + RED_STATUS_STR[status];
  return std::string("fpLLL reduction failed with status ") + std::to_string(status);
}

size_t checked_entry_count(int rows, int cols)
{
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("an integer matrix must have positive dimensions");
  const size_t r = static_cast<size_t>(rows);
  const size_t c = static_cast<size_t>(cols);
  if (r > std::numeric_limits<size_t>::max() / c)
    throw std::invalid_argument("integer matrix has too many entries");
  return r * c;
}

void copy_input_matrix(ZZ_mat<mpz_t> &matrix,
                       const void *const *entries,
                       size_t entry_count)
{
  const int rows = matrix.get_rows();
  const int cols = matrix.get_cols();
  const size_t expected = checked_entry_count(rows, cols);
  if (entry_count != expected)
    throw std::invalid_argument("integer matrix entry count does not match its dimensions");
  if (entries == nullptr)
    throw std::invalid_argument("integer matrix mpz pointer array is null");

  size_t k = 0;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j, ++k)
    {
      if (entries[k] == nullptr)
        throw std::invalid_argument("integer matrix contains a null mpz pointer");
      const mpz_srcptr source = reinterpret_cast<mpz_srcptr>(entries[k]);
      mpz_set(matrix[i][j].get_data(), source);
    }
  }
}

void validate_common(double delta, double eta, int float_type, int precision, int flags)
{
  if (!(delta > 0.25 && delta < 1.0))
    throw std::invalid_argument("delta must satisfy 0.25 < delta < 1");
  if (!(eta >= 0.5 && eta < std::sqrt(delta)))
    throw std::invalid_argument("eta must satisfy 0.5 <= eta < sqrt(delta)");
  if (float_type < FT_DEFAULT || float_type > FT_MPFR)
    throw std::invalid_argument("unknown fpLLL floating-point type");
  if (precision < 0)
    throw std::invalid_argument("precision must be nonnegative");
  if ((flags & ~(LLL_VERBOSE | LLL_EARLY_RED | LLL_SIEGEL)) != 0)
    throw std::invalid_argument("unknown fpLLL LLL flag");

  if (precision != 0 && float_type != FT_DEFAULT && float_type != FT_MPFR)
    throw std::invalid_argument("an explicit precision requires MPFR floating point");

#ifndef FPLLL_WITH_LONG_DOUBLE
  if (float_type == FT_LONG_DOUBLE)
    throw std::invalid_argument("this fpLLL build has no long-double support");
#endif
#ifndef FPLLL_WITH_DPE
  if (float_type == FT_DPE)
    throw std::invalid_argument("this fpLLL build has no DPE support");
#endif
#ifndef FPLLL_WITH_QD
  if (float_type == FT_DD || float_type == FT_QD)
    throw std::invalid_argument("this fpLLL build has no QD support");
#endif
}

void validate_basis_options(int method, int float_type, int precision, int flags)
{
  if (method < LM_WRAPPER || method > LM_FAST)
    throw std::invalid_argument("unknown fpLLL LLL method");

  if (method == LM_WRAPPER)
  {
    if (float_type != FT_DEFAULT)
      throw std::invalid_argument("the wrapper method requires float_type=:default");
    if (precision != 0)
      throw std::invalid_argument("the wrapper method does not accept an explicit precision");
  }

  if (method == LM_PROVED && (flags & LLL_EARLY_RED) != 0)
    throw std::invalid_argument("proved LLL with early reduction is not implemented by fpLLL");

  if (method == LM_FAST &&
      float_type != FT_DEFAULT &&
      float_type != FT_DOUBLE &&
      float_type != FT_LONG_DOUBLE &&
      float_type != FT_DD &&
      float_type != FT_QD)
    throw std::invalid_argument("fast LLL requires double, long double, double-double, or quad-double");
}

void validate_square_symmetric(const ZZ_mat<mpz_t> &gram)
{
  if (gram.get_rows() != gram.get_cols())
    throw std::invalid_argument("a Gram matrix must be square");
  for (int i = 0; i < gram.get_rows(); ++i)
    for (int j = 0; j < i; ++j)
      if (gram[i][j] != gram[j][i])
        throw std::invalid_argument("a Gram matrix must be symmetric");
}

template <class F>
int reduce_gram_typed(ZZ_mat<mpz_t> &gram,
                      ZZ_mat<mpz_t> &transform,
                      ZZ_mat<mpz_t> &inverse_transform_transpose,
                      double delta,
                      double eta,
                      int flags)
{
  using ZT = Z_NR<mpz_t>;
  using FT = FP_NR<F>;

  MatGSOGram<ZT, FT> gso(gram, transform, inverse_transform_transpose, GSO_INT_GRAM);
  if (!gso.update_gso())
    return RED_GSO_FAILURE;
  LLLReduction<ZT, FT> reduction(gso, delta, eta, flags);
  reduction.lll();
  return reduction.status;
}

int reduce_gram(ZZ_mat<mpz_t> &gram,
                ZZ_mat<mpz_t> &transform,
                ZZ_mat<mpz_t> &inverse_transform_transpose,
                double delta,
                double eta,
                int float_type,
                int precision,
                int flags)
{
  int selected = float_type == FT_DEFAULT ? FT_MPFR : float_type;

  switch (selected)
  {
  case FT_DOUBLE:
    return reduce_gram_typed<double>(gram, transform, inverse_transform_transpose,
                                     delta, eta, flags);
#ifdef FPLLL_WITH_LONG_DOUBLE
  case FT_LONG_DOUBLE:
    return reduce_gram_typed<long double>(gram, transform, inverse_transform_transpose,
                                          delta, eta, flags);
#endif
#ifdef FPLLL_WITH_DPE
  case FT_DPE:
    return reduce_gram_typed<dpe_t>(gram, transform, inverse_transform_transpose,
                                    delta, eta, flags);
#endif
#ifdef FPLLL_WITH_QD
  case FT_DD:
  case FT_QD:
  {
    unsigned int old_control_word;
    fpu_fix_start(&old_control_word);
    try
    {
      int status = selected == FT_DD
                       ? reduce_gram_typed<dd_real>(gram, transform,
                                                   inverse_transform_transpose,
                                                   delta, eta, flags)
                       : reduce_gram_typed<qd_real>(gram, transform,
                                                   inverse_transform_transpose,
                                                   delta, eta, flags);
      fpu_fix_end(&old_control_word);
      return status;
    }
    catch (...)
    {
      fpu_fix_end(&old_control_word);
      throw;
    }
  }
#endif
  case FT_MPFR:
  {
    const int old_precision = FP_NR<mpfr_t>::get_prec();
    const int selected_precision =
        precision == 0 ? l2_min_prec(gram.get_rows(), delta, eta, LLL_DEF_EPSILON) : precision;
    FP_NR<mpfr_t>::set_prec(selected_precision);
    try
    {
      int status = reduce_gram_typed<mpfr_t>(gram, transform,
                                             inverse_transform_transpose,
                                             delta, eta, flags);
      FP_NR<mpfr_t>::set_prec(old_precision);
      return status;
    }
    catch (...)
    {
      FP_NR<mpfr_t>::set_prec(old_precision);
      throw;
    }
  }
  default:
    throw std::invalid_argument("unsupported floating-point type for Gram reduction");
  }
}

const ZZ_mat<mpz_t> *select_matrix(const fplll_julia_result *result, int which)
{
  if (result == nullptr)
    return nullptr;
  switch (which)
  {
  case FPLLL_JULIA_RESULT_PRIMARY:
    return &result->primary;
  case FPLLL_JULIA_RESULT_TRANSFORM:
    return result->has_transform ? &result->transform : nullptr;
  case FPLLL_JULIA_RESULT_INVERSE:
    return result->has_inverse ? &result->inverse : nullptr;
  default:
    return nullptr;
  }
}

} // namespace

extern "C" uint32_t fplll_julia_capabilities(void)
{
  uint32_t result = FPLLL_JULIA_CAP_MPFR;
#ifdef FPLLL_WITH_LONG_DOUBLE
  result |= FPLLL_JULIA_CAP_LONG_DOUBLE;
#endif
#ifdef FPLLL_WITH_DPE
  result |= FPLLL_JULIA_CAP_DPE;
#endif
#ifdef FPLLL_WITH_QD
  result |= FPLLL_JULIA_CAP_QD;
#endif
  return result;
}

extern "C" uint32_t fplll_julia_gmp_limb_bits(void)
{
  return static_cast<uint32_t>(sizeof(mp_limb_t) * CHAR_BIT);
}

extern "C" uint32_t fplll_julia_gmp_nail_bits(void)
{
  return static_cast<uint32_t>(GMP_NAIL_BITS);
}

extern "C" int fplll_julia_min_precision(int dimension, double delta, double eta)
{
  if (dimension <= 0 || !(delta > 0.25 && delta < 1.0) ||
      !(eta >= 0.5 && eta < std::sqrt(delta)))
    return -1;
  try
  {
    return l2_min_prec(dimension, delta, eta, LLL_DEF_EPSILON);
  }
  catch (...)
  {
    return -1;
  }
}

extern "C" int fplll_julia_lll_mpz(
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
    char **error_out)
{
  clear_outputs(result_out, error_out);
  if (result_out == nullptr)
  {
    set_error(error_out, "result output pointer is null");
    return -1;
  }

  try
  {
    validate_common(delta, eta, float_type, precision, flags);
    validate_basis_options(method, float_type, precision, flags);
    checked_entry_count(rows, cols);

    std::unique_ptr<fplll_julia_result> result(new fplll_julia_result(rows, cols));
    copy_input_matrix(result->primary, entries, entry_count);

    const bool compute_transform = want_transform != 0 || want_inverse_transform != 0;
    if (compute_transform)
      result->transform.gen_identity(rows);
    if (want_inverse_transform != 0)
      result->inverse.gen_identity(rows);

    int status = lll_reduction(result->primary, result->transform, result->inverse,
                               delta, eta,
                               static_cast<LLLMethod>(method),
                               static_cast<FloatType>(float_type),
                               precision, flags);
    if (status != RED_SUCCESS)
    {
      set_error(error_out, status_message(status));
      return status;
    }

    result->has_transform = want_transform != 0;
    result->has_inverse = want_inverse_transform != 0;
    *result_out = result.release();
    return 0;
  }
  catch (const std::exception &error)
  {
    set_error(error_out, error.what());
    return -1;
  }
  catch (...)
  {
    set_error(error_out, "unknown C++ exception in the fpLLL Julia shim");
    return -1;
  }
}

extern "C" int fplll_julia_lll_gram_mpz(
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
    char **error_out)
{
  clear_outputs(result_out, error_out);
  if (result_out == nullptr)
  {
    set_error(error_out, "result output pointer is null");
    return -1;
  }

  try
  {
    validate_common(delta, eta, float_type, precision, flags);
    if ((flags & LLL_EARLY_RED) != 0)
      throw std::invalid_argument("early reduction is not supported by the Gram-matrix interface");
    if (rows != cols)
      throw std::invalid_argument("a Gram matrix must be square");
    checked_entry_count(rows, cols);

    std::unique_ptr<fplll_julia_result> result(new fplll_julia_result(rows, cols));
    copy_input_matrix(result->primary, entries, entry_count);
    validate_square_symmetric(result->primary);

    const bool compute_transform = want_transform != 0 || want_inverse_transform != 0;
    if (compute_transform)
      result->transform.gen_identity(rows);
    if (want_inverse_transform != 0)
      result->inverse.gen_identity(rows);

    int status = reduce_gram(result->primary, result->transform, result->inverse,
                             delta, eta, float_type, precision, flags);
    if (status != RED_SUCCESS)
    {
      set_error(error_out, status_message(status));
      return status;
    }

    result->has_transform = want_transform != 0;
    result->has_inverse = want_inverse_transform != 0;
    if (result->has_inverse)
    {
      /* MatGSO stores U^{-T}; Julia users expect U^{-1}. */
      result->inverse.transpose();
    }

    *result_out = result.release();
    return 0;
  }
  catch (const std::exception &error)
  {
    set_error(error_out, error.what());
    return -1;
  }
  catch (...)
  {
    set_error(error_out, "unknown C++ exception in the fpLLL Gram Julia shim");
    return -1;
  }
}

extern "C" int fplll_julia_result_shape(
    const fplll_julia_result *result,
    int which,
    int *rows_out,
    int *cols_out)
{
  if (rows_out == nullptr || cols_out == nullptr)
    return -1;
  const ZZ_mat<mpz_t> *matrix = select_matrix(result, which);
  if (matrix == nullptr)
    return -2;
  *rows_out = matrix->get_rows();
  *cols_out = matrix->get_cols();
  return 0;
}

extern "C" int fplll_julia_result_entry_ptrs(
    const fplll_julia_result *result,
    int which,
    const void **entries_out,
    size_t capacity)
{
  const ZZ_mat<mpz_t> *matrix = select_matrix(result, which);
  if (matrix == nullptr)
    return -2;
  const int rows = matrix->get_rows();
  const int cols = matrix->get_cols();
  if (rows <= 0 || cols <= 0)
    return -4;
  const size_t r = static_cast<size_t>(rows);
  const size_t c = static_cast<size_t>(cols);
  if (r > std::numeric_limits<size_t>::max() / c)
    return -4;
  const size_t count = r * c;
  if (capacity != count)
    return -3;
  if (entries_out == nullptr)
    return -1;

  size_t k = 0;
  for (int i = 0; i < matrix->get_rows(); ++i)
    for (int j = 0; j < matrix->get_cols(); ++j, ++k)
      entries_out[k] = static_cast<const void *>((*matrix)[i][j].get_data());
  return 0;
}

extern "C" void fplll_julia_result_free(fplll_julia_result *result)
{
  delete result;
}

extern "C" void fplll_julia_free(void *ptr)
{
  std::free(ptr);
}
