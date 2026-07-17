#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/check_diagonal_coordinate_scaling.cpp" <<'CPP'
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "dev_src/diagonal_coordinate_scaling.h"

int main()
{
    const int count = 4;
    const double physical[count] = {2.0, -0.25, 4.5, -3.0};
    const double reference[count] = {1.75, -1.0, 4.5, 2.0};
    const double scale[count] = {0.2, 3.0, 11.0, 0.75};
    double coordinate[count] = {};

    for (int i = 0; i < count; ++i) {
        coordinate[i] =
            sus2::DiagonalCoordinateScaling::PackCentered(
                physical[i], reference[i], scale[i]);
        const double reconstructed =
            sus2::DiagonalCoordinateScaling::UnpackCentered(
                coordinate[i], reference[i], scale[i]);
        if (reconstructed != physical[i]) {
            std::cerr << "coordinate round trip changed physical value at "
                      << i << std::endl;
            return 1;
        }
    }

    auto loss = [&](const double* values) {
        double result = 0.0;
        for (int i = 0; i < count; ++i)
            result += 0.5 * (i + 1.0) * values[i] * values[i]
                    + std::sin(values[i]);
        return result;
    };

    const double step = 1.0e-6;
    for (int varied = 0; varied < count; ++varied) {
        double plus[count] = {};
        double minus[count] = {};
        for (int i = 0; i < count; ++i) {
            double z_plus = coordinate[i];
            double z_minus = coordinate[i];
            if (i == varied) {
                z_plus += step;
                z_minus -= step;
            }
            plus[i] =
                sus2::DiagonalCoordinateScaling::UnpackCentered(
                    z_plus, reference[i], scale[i]);
            minus[i] =
                sus2::DiagonalCoordinateScaling::UnpackCentered(
                    z_minus, reference[i], scale[i]);
        }
        const double finite_difference =
            (loss(plus) - loss(minus)) / (2.0 * step);
        const double physical_gradient =
            (varied + 1.0) * physical[varied] + std::cos(physical[varied]);
        const double pulled_back =
            sus2::DiagonalCoordinateScaling::PullbackGradient(
                physical_gradient, scale[varied]);
        const double comparison_scale =
            std::max(1.0, std::abs(finite_difference));
        if (std::abs(finite_difference - pulled_back)
            > 2.0e-8 * comparison_scale) {
            std::cerr << "coordinate gradient mismatch at " << varied
                      << " finite_difference=" << finite_difference
                      << " pulled_back=" << pulled_back << std::endl;
            return 1;
        }
    }

    const double low_block_median = 0.2;
    const double floor = 1.0e-6;
    const double median_scale =
        sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
            low_block_median, low_block_median, floor);
    const double stiff_scale =
        sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
            0.8, low_block_median, floor);
    const double floored_scale =
        sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
            0.0, low_block_median, floor);
    const double high_block_scale =
        sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
            4.0, 4.0, floor);
    if (std::abs(median_scale - 1.0) > 1.0e-14
        || std::abs(stiff_scale - 2.0) > 1.0e-14
        || std::abs(floored_scale - 1.0) > 1.0e-14
        || std::abs(high_block_scale - 2.0) > 1.0e-14) {
        std::cerr << "curvature scale stabilization mismatch" << std::endl;
        return 1;
    }

    const double tiny_scale =
        sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
            1.0e-20, 10.0, floor);
    if (tiny_scale < 1.0 || 1.0 / (tiny_scale * tiny_scale) > 1.0) {
        std::cerr << "coordinate scale amplified the physical BFGS step"
                  << std::endl;
        return 1;
    }

    return 0;
}
CPP

g++ -std=c++11 -I"$ROOT" \
  "$TMPDIR/check_diagonal_coordinate_scaling.cpp" \
  -o "$TMPDIR/check_diagonal_coordinate_scaling"
"$TMPDIR/check_diagonal_coordinate_scaling"

echo "diagonal_coordinate_scaling_check passed"
