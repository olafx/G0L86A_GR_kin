#pragma once

#include "util.hpp"

template <size_t N>
using Vec = util::Vec<double, N>;

template <size_t N1, size_t N2>
using Mat = util::Ten<double, N1, N2>::V;

using int2 = util::Vec<int, 2>;
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec3_sph = Vec3;
using Vec3_Car = Vec3;
using RGB = util::Vec<uint8_t, 3>;
using Vec6 = Vec<6>;
using Mat3 = Mat<3, 3>;
using Mat23 = Mat<3, 2>;
using Ten3 = util::Ten<double, 3, 3, 3>::V;
