#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <array>
#include <concepts>
#include <format>
#include <mdspan>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// 
// Compile-time size arrays similar to those in GLM, with anonymous union
// fields.
// Benefits over std::array:
// - Due to the anonymous union we can write vec.L, vec.y, etc.
// - Nested initialized works, e.g. `{{1, 2}, {3, 4}}` for a
//   `Vec<Vec<int, 2>, 2>`. This does not work for nested `std::array`.
// - Nested formatting works as one would want it to. It does not work at all
//   for `std::array`.
// There's a small utility `Ten` class to produce tensors, e.g.
// `Ten<int, 2, 2>::V` is `Vec<Vec<int, 2>, 2>`. So it's a 'type factory', not
// an actual type.
// 
// NOTE: Can reinterpret cast these as is convenient, at least for
//   `is_trivially_default_constructible` underlying types I suppose.
// 
////////////////////////////////////////////////////////////////////////////////

namespace util
{

template <typename A, size_t N>
requires std::is_trivially_copyable_v<A>
struct Vec;

namespace detail
{

template <typename T>
struct FlatTraits
{ using Scalar = T;
  static constexpr size_t size = 1;
};

template <typename A, typename B>
concept ReshapeCompatible =
  std::same_as<
    std::remove_cv_t<typename FlatTraits<A>::Scalar>,
    std::remove_cv_t<typename FlatTraits<B>::Scalar>
  > &&
  (FlatTraits<A>::size == FlatTraits<B>::size) &&
  (sizeof(A) == sizeof(B)) &&
  std::is_trivially_copyable_v<A> &&
  std::is_trivially_copyable_v<B>;

template <typename A, typename B>
requires ReshapeCompatible<A, B>
constexpr A reshape_cast
( const B& x
)
{ return std::bit_cast<A>(x);
}

template <typename A, size_t... Ns>
struct WrapTen;

template <typename A>
struct WrapTen<A>
{ using V = A;
};

template <typename A, size_t N, size_t... Ns>
struct WrapTen<A, N, Ns...>
{ using V = typename WrapTen<Vec<A, N>, Ns...>::V;
};

} // namespace detail

template <typename A, size_t N, size_t... Ns>
struct Ten
{ using V = typename detail::WrapTen<A, N, Ns...>::V;
  Ten() = delete;
  Ten(const Ten&) = delete;
  Ten& operator=(const Ten&) = delete;
};

////////////////////////////////////////////////////////////////////////////////

template <typename A>
requires std::is_trivially_copyable_v<A>
struct Vec<A, 2>
{
  using Scalar = typename detail::FlatTraits<A>::Scalar;
  static constexpr size_t size = 2*detail::FlatTraits<A>::size;

  union
  { struct { A x, y; };
    struct { A u, v; };
    struct { A a, b; };
    struct { A m, p; };
    struct { A i, j; };
    struct { A L, R; };
    struct { A X, V; };
    std::array<A, 2> data;
  };

  constexpr Vec() = default;

  constexpr Vec
  ( A x, A y
  ) : x(x), y(y)
  {}

  template <typename B>
  requires std::is_convertible_v<B, A>
  constexpr Vec
  ( const Vec<B, 2>& other
  )
  { for (size_t i = 0; i < 2; i++)
      data[i] = static_cast<A>(other.data[i]);
  }

  template <typename B, size_t M>
  requires detail::ReshapeCompatible<Vec<A, 2>, Vec<B, M>> &&
    (!(M == 2 && std::is_convertible_v<B, A>))
  constexpr Vec
  ( const Vec<B, M>& other
  )
  { *this = detail::reshape_cast<Vec>(other);
  }

  constexpr A& operator[]
  ( auto i
  ) noexcept
  { return data[i];
  }
  
  constexpr A const& operator[]
  ( auto i
  ) const noexcept
  { return data[i];
  }
}; // struct Vec<A, 2>

template <typename A>
requires std::is_trivially_copyable_v<A>
struct Vec<A, 3>
{
  using Scalar = typename detail::FlatTraits<A>::Scalar;
  static constexpr size_t size = 3*detail::FlatTraits<A>::size;

  union
  { struct { A x, y, z; };
    struct { A r, th, phi; };
    struct { A u, v, w; };
    struct { A a, b, c; };
    struct { A i, j, k; };
    struct { A R, G, B; };
    std::array<A, 3> data;
  };

  constexpr Vec() = default;

  constexpr Vec
  ( A x, A y, A z
  ) : x(x), y(y), z(z)
  {}

  template <typename B>
  requires std::is_convertible_v<B, A>
  constexpr Vec
  ( const Vec<B, 3>& other
  )
  { for (size_t i = 0; i < 3; i++)
      data[i] = static_cast<A>(other.data[i]);
  }

  template <typename B, size_t M>
  requires detail::ReshapeCompatible<Vec<A, 3>, Vec<B, M>> &&
    (!(M == 3 && std::is_convertible_v<B, A>))
  constexpr Vec
  ( const Vec<B, M>& other
  )
  { *this = detail::reshape_cast<Vec>(other);
  }

  constexpr A& operator[]
  ( auto i
  ) noexcept
  { return data[i];
  }

  constexpr A const& operator[]
  ( auto i
  ) const noexcept
  { return data[i];
  }
}; // struct Vec<A, 3>

template <typename T>
requires std::is_trivially_copyable_v<T>
struct Vec<T, 4>
{
  using Scalar = typename detail::FlatTraits<T>::Scalar;
  static constexpr size_t size = 4*detail::FlatTraits<T>::size;

  union
  { struct { T x, y, z, w; };
    struct { T a, b, c, d; };
    struct { T i, j, k, l; };
    struct { T R, G, B, A; };
    std::array<T, 4> data;
  };

  constexpr Vec() = default;

  constexpr Vec
  ( T x, T y, T z, T w
  ) : x(x), y(y), z(z), w(w)
  {}

  template <typename B>
  requires std::is_convertible_v<B, T>
  constexpr Vec
  ( const Vec<B, 4>& other
  )
  { for (size_t i = 0; i < 4; i++)
      data[i] = static_cast<T>(other.data[i]);
  }

  template <typename B, size_t M>
  requires detail::ReshapeCompatible<Vec<T, 4>, Vec<B, M>> &&
    (!(M == 4 && std::is_convertible_v<B, T>))
  constexpr Vec
  ( const Vec<B, M>& other
  )
  { *this = detail::reshape_cast<Vec>(other);
  }

  constexpr T& operator[]
  ( auto i
  ) noexcept
  { return data[i];
  }

  constexpr T const& operator[]
  ( auto i
  ) const noexcept
  { return data[i];
  }
}; // struct Vec<T, 4>

template <typename A, size_t N>
requires std::is_trivially_copyable_v<A>
struct Vec
{
  using Scalar = typename detail::FlatTraits<A>::Scalar;
  static constexpr size_t size = N*detail::FlatTraits<A>::size;

  std::array<A, N> data;

  constexpr Vec() = default;

  template <typename... B>
  requires
    (sizeof...(B) == N) &&
    (std::is_convertible_v<B, A> && ...)
  constexpr explicit Vec
  ( B&&... args
  ) : data { static_cast<A>(std::forward<B>(args))... }
  {}

  template <typename B>
  requires std::is_convertible_v<B, A>
  constexpr Vec
  ( Vec<B, N> const& other
  )
  { for (size_t i = 0; i < N; i++)
      data[i] = static_cast<A>(other.data[i]);
  }

  template <typename B, size_t M>
  requires detail::ReshapeCompatible<Vec<A, N>, Vec<B, M>> &&
    (!(M == N && std::is_convertible_v<B, A>))
  constexpr Vec
  ( const Vec<B, M>& other
  )
  { *this = detail::reshape_cast<Vec>(other);
  }

  constexpr A& operator[]
  ( auto i
  ) noexcept
  { return data[i];
  }

  constexpr A const& operator[]
  ( auto i
  ) const noexcept
  { return data[i];
  }
}; // struct Vec<A, N>

template <typename A, size_t N>
struct detail::FlatTraits<Vec<A, N>>
{ using Scalar = typename FlatTraits<A>::Scalar;
  static constexpr size_t size = N*FlatTraits<A>::size;
};

////////////////////////////////////////////////////////////////////////////////

// vector + vector
template <typename A, size_t N, typename B>
requires requires (A const& a, B const& b) { a+b; }
constexpr auto operator+
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()+std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]+rhs[i]);
  return out;
}

// vector - vector
template <typename A, size_t N, typename B>
requires requires (A const& a, B const& b) { a-b; }
constexpr auto operator-
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()-std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]-rhs[i]);
  return out;
}

// vector + scalar
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { a+b; }
constexpr auto operator+
( Vec<A, N> const& lhs,
  B const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()+std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]+rhs);
  return out;
}

// scalar + vector
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (Vec<A, N> const& v, B const& b) { v+b; }
constexpr auto operator+
( B const& lhs,
  Vec<A, N> const& rhs
)
{ return rhs+lhs;
}

// vector - scalar
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { a-b; }
constexpr auto operator-
( Vec<A, N> const& lhs,
  B const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()-std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]-rhs);
  return out;
}

// scalar - vector
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { b-a; }
constexpr auto operator-
( B const& lhs,
  Vec<A, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<B const&>()-std::declval<A const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs-rhs[i]);
  return out;
}

// vector * vector
template <typename A, size_t N, typename B>
requires requires (A const& a, B const& b) { a*b; }
constexpr auto operator*
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()*std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]*rhs[i]);
  return out;
}

// vector * scalar
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { a*b; }
constexpr auto operator*
( Vec<A, N> const& lhs,
  B const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()*std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]*rhs);
  return out;
}

// scalar * vector
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (Vec<A, N> const& v, B const& b) { v*b; }
constexpr auto operator*
( B const& lhs,
  Vec<A, N> const& rhs
)
{ return rhs*lhs;
}

// vector / vector
template <typename A, size_t N, typename B>
requires requires (A const& a, B const& b) { a/b; }
constexpr auto operator/
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()/std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]/rhs[i]);
  return out;
}

// vector / scalar
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { a/b; }
constexpr auto operator/
( Vec<A, N> const& lhs,
  B const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()/std::declval<B const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs[i]/rhs);
  return out;
}

// scalar / vector
template <typename A, size_t N, typename B>
requires (!std::is_convertible_v<B, Vec<A, N>>) &&
requires (A const& a, B const& b) { b/a; }
constexpr auto operator/
( B const& lhs,
  Vec<A, N> const& rhs
)
{ using R = std::remove_cvref_t<
    decltype(std::declval<B const&>()/std::declval<A const&>())>;
  Vec<R, N> out {};
  for (size_t i = 0; i < N; i++)
    out[i] = static_cast<R>(lhs/rhs[i]);
  return out;
}

// vector += rhs
template <typename A, size_t N, typename B>
requires requires (Vec<A, N>& l, B const& r) { l = l+r; }
constexpr Vec<A, N>& operator+=
( Vec<A, N>& lhs,
  B const& rhs
)
{ lhs = lhs+rhs;
  return lhs;
}

// vector -= rhs
template <typename A, size_t N, typename B>
requires requires (Vec<A, N>& l, B const& r) { l = l-r; }
constexpr Vec<A, N>& operator-=
( Vec<A, N>& lhs,
  B const& rhs
)
{ lhs = lhs-rhs;
  return lhs;
}

// vector *= rhs
template <typename A, size_t N, typename B>
requires requires (Vec<A, N>& l, B const& r) { l = l*r; }
constexpr Vec<A, N>& operator*=
( Vec<A, N>& lhs,
  B const& rhs
)
{ lhs = lhs*rhs;
  return lhs;
}

// vector /= rhs
template <typename A, size_t N, typename B>
requires requires (Vec<A, N>& l, B const& r) { l = l/r; }
constexpr Vec<A, N>& operator/=
( Vec<A, N>& lhs,
  B const& rhs
)
{ lhs = lhs/rhs;
  return lhs;
}

// vector == vector
template <typename A, size_t N, typename B>
requires requires (A const& a, B const& b)
{ { a == b } -> std::convertible_to<bool>;
}
constexpr bool operator==
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ for (size_t i = 0; i < N; i++)
    if (!(lhs[i] == rhs[i]))
      return false;
  return true;
}

// vector != vector
template <typename A, size_t N, typename B>
requires requires (Vec<A, N> const& l, Vec<B, N> const& r)
{ { l == r } -> std::convertible_to<bool>;
}
constexpr bool operator!=
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ return !(lhs == rhs);
}

// dot product
template <std::floating_point A, size_t N, std::floating_point B>
constexpr auto dot
( Vec<A, N> const& lhs,
  Vec<B, N> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()*std::declval<B const&>())>;
  R out {};
  for (size_t i = 0; i < N; i++)
    out += static_cast<R>(lhs[i]*rhs[i]);
  return out;
}

// cross product
template <std::floating_point A, std::floating_point B>
constexpr auto cross
( Vec<A, 3> const& lhs,
  Vec<B, 3> const& rhs
)
{ using R = std::remove_cvref_t<decltype(
    std::declval<A const&>()*std::declval<B const&>())>;
  return Vec<R, 3>
  { static_cast<R>(lhs[1]*rhs[2]-lhs[2]*rhs[1]),
    static_cast<R>(lhs[2]*rhs[0]-lhs[0]*rhs[2]),
    static_cast<R>(lhs[0]*rhs[1]-lhs[1]*rhs[0])
  };
}

// squared norm
template <std::floating_point A, size_t N>
constexpr auto norm2
( const Vec<A, N>& vec
)
{ return dot(vec, vec);
}

// norm
template <std::floating_point A, size_t N>
constexpr auto norm
( const Vec<A, N>& vec
)
{ return std::sqrt(norm2(vec));
}

// normalized
template <std::floating_point A, size_t N>
constexpr auto normalized
( const Vec<A, N>& vec
)
{ return vec/norm(vec);
}

// normalize
template <std::floating_point A, size_t N>
constexpr void normalize
( Vec<A, N>& vec
)
{ vec /= norm(vec);
}

} // namespace util

////////////////////////////////////////////////////////////////////////////////

namespace std
{

// NOTE: Recursive util::Vec will print as one would desire, also with
//   formatting for the underlying, e.g. `"{::.2e}"` for a
//   `Vec<Vec<int, 3>, 2>`.

template <typename A, size_t N, typename CharT>
struct formatter<util::Vec<A, N>, CharT>
{
  std::basic_string_view<CharT> spec {};

  constexpr auto parse(std::basic_format_parse_context<CharT>& ctx)
  { const auto begin = ctx.begin();
    const auto end = ctx.end();
    auto it = begin;
    for (; it != end && *it != CharT('}'); it++);
    spec = { begin, static_cast<size_t>(it-begin) };
    return it;
  }

  template <typename FormatContext>
  auto format(const util::Vec<A, N>& vec, FormatContext& ctx) const
  { static_assert
    ( std::is_same_v<CharT, char> ||
      std::is_same_v<CharT, wchar_t>
    );
// NOTE: Reconstruct `"{:<spec>}"` or `"{:<spec>}"`
    std::basic_string<CharT> fmt;
    if (spec.empty())
      fmt = {CharT('{'), CharT('}')};
    else
    { const bool implicit_elem_spec = spec.front() != CharT(':');
      fmt.reserve(spec.size()+(implicit_elem_spec ? 4 : 3));
      fmt.push_back(CharT('{'));
      fmt.push_back(CharT(':'));
      if (implicit_elem_spec)
        fmt.push_back(CharT(':'));
      fmt.append(spec);
      fmt.push_back(CharT('}'));
    }
    if constexpr (std::is_same_v<CharT, char>)
      return std::vformat_to(ctx.out(), fmt, std::make_format_args(vec.data));
    else
      return std::vformat_to(ctx.out(), fmt, std::make_wformat_args(vec.data));
  }
};


} // namespace std

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Coordinate conversions.
// 
////////////////////////////////////////////////////////////////////////////////

namespace util
{

template <std::floating_point T>
void sph_to_Car
( Vec<T, 3> r,
  Vec<T, 3>& x
)
{ x.x = r.r*sin(r.th)*cos(r.phi);
  x.y = r.r*sin(r.th)*sin(r.phi);
  x.z = r.r*cos(r.th);
}

template <std::floating_point T>
Vec<T, 3> sph_to_Car
( Vec<T, 3> x_sph
)
{ Vec<T, 3> x_Car;
  sph_to_Car(x_sph, x_Car);
  return x_Car;
}

template <std::floating_point T>
void Car_to_sph
( Vec<T, 3> x,
  Vec<T, 3>& r
)
{ r.r = norm(x);
  r.th = acos(x.z/r.r);
  r.phi = atan2(x.y, x.x);
}

template <std::floating_point T>
Vec<T, 3> Car_to_sph
( Vec<T, 3> x_Car
)
{ Vec<T, 3> x_sph;
  Car_to_sph(x_Car, x_sph);
  return x_sph;
}

// Spherical coordinate basis vectors evaluated in spherical coordinates
// and represented in Cartesian coordinates.
template <std::floating_point T>
Ten<T, 3, 3>::V e_sph
( Vec<T, 3> r
)
{ return
  { {sin(r.th)*cos(r.phi), sin(r.th)*sin(r.phi),  cos(r.th)},
    {cos(r.th)*cos(r.phi), cos(r.th)*sin(r.phi), -sin(r.th)},
    {         -sin(r.phi),           cos(r.phi),          0},
  };
}

} // namespace util

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// 
// Conversion to pybind11 objects.
// 
////////////////////////////////////////////////////////////////////////////////

namespace util
{

namespace py = pybind11;

// A recursive scheme to construct pybind11 shapes and strides from
// `Ten<T, ...>:V`.
namespace detail
{

// TODO: There should be a way to just get the n-th template parameter now. This
// kind of style of template metaprogramming can be improved now I believe.

template <typename T>
struct PyArrayTraits
{ using ScalarT = T;
  static constexpr size_t rank = 0;

  template <size_t rank_>
  static constexpr void fill
  ( std::array<py::ssize_t, rank_>&,
    std::array<py::ssize_t, rank_>&,
    size_t
  )
  {}
};

template <typename A, size_t N>
struct PyArrayTraits<Vec<A, N>>
{ using Traits = PyArrayTraits<A>;
  using ScalarT = typename Traits::ScalarT;
  static constexpr size_t rank = 1+Traits::rank;

  template <size_t rank_>
  static constexpr void fill
  ( std::array<py::ssize_t, rank_>& shape,
    std::array<py::ssize_t, rank_>& strides,
    size_t off
  )
  { shape  [off] = static_cast<py::ssize_t>(N);
    strides[off] = static_cast<py::ssize_t>(sizeof(A));
    Traits::fill(shape, strides, off+1);
  }
};

} // namespace detail

// For `std::vector<Vec<...>>`, `Vec` treated recursively.
// 
// NOTE: Ownership of `x` is lost.
template <typename Vec>
py::array_t<typename detail::PyArrayTraits<Vec>::ScalarT>
to_py_array
( std::vector<Vec>&& x
)
{ using Traits = detail::PyArrayTraits<Vec>;
  using ScalarT = typename Traits::ScalarT;
  constexpr size_t rank = 1+Traits::rank;
  std::array<py::ssize_t, rank> shape {}, strides {};
  shape  [0] = static_cast<py::ssize_t>(x.size());
  strides[0] = static_cast<py::ssize_t>(sizeof(Vec));
  Traits::fill(shape, strides, 1);
  auto* x_ = new std::vector<Vec>(std::move(x));
  auto owner = py::capsule {x_,
    [](void* p) { delete static_cast<decltype(x_)>(p); }
  };
  return {shape, strides, reinterpret_cast<ScalarT*>(x_->data()), owner};
}

// For `std::mdspan<Vec<...>>`, `Vec` treated recursively.
// 
// NOTE: Ownership of `x.data_handle()` is lost.
// NOTE: `x.data_handle()` must have been allocated via `new[]`.
// NOTE: Does not support every layout and access policy.
template <typename Vec, typename Extents>
requires (!std::is_const_v<Vec>)
py::array_t<typename detail::PyArrayTraits<Vec>::ScalarT>
to_py_array
( std::mdspan<Vec, Extents, std::layout_right> x
)
{ using Traits = detail::PyArrayTraits<Vec>;
  using ScalarT = typename Traits::ScalarT;
  constexpr size_t rank = Extents::rank()+Traits::rank;
  std::array<py::ssize_t, rank> shape {}, strides {};
  auto stride = static_cast<py::ssize_t>(sizeof(Vec));
  for (size_t i = Extents::rank(); i-- > 0;)
  { shape[i] = static_cast<py::ssize_t>(x.extent(i));
    strides[i] = stride;
    stride *= shape[i];
  }
  Traits::fill(shape, strides, Extents::rank());
  auto* data = x.data_handle();
  auto owner = py::capsule {data,
    [](void* p) { delete[] static_cast<Vec*>(p); }
  };
  return {shape, strides, reinterpret_cast<ScalarT*>(data), owner};
}

} // namespace util
