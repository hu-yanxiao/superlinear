#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/check_force_loss.cpp" <<'CPP'
#include <cmath>
#include <iostream>
#include <string>

#include "dev_src/force_loss.h"

namespace {
void require_close(double actual, double expected, double tol, const std::string& label)
{
    if (std::abs(actual - expected) > tol) {
        std::cerr << label << " actual=" << actual
                  << " expected=" << expected << std::endl;
        std::exit(1);
    }
}
}

int main()
{
    const double scale = 2.0;
    require_close(ForceResidualLoss(3.0, ForceLossKind::L2, scale),
                  9.0, 1.0e-14, "l2 loss");
    require_close(ForceResidualGrad(3.0, ForceLossKind::L2, scale),
                  6.0, 1.0e-14, "l2 grad");
    require_close(ForceResidualCurvature(3.0, ForceLossKind::L2, scale),
                  2.0, 1.0e-14, "l2 curvature");

    const double residual = 4.0;
    const double x = residual / scale;
    const double expected_loss = scale * scale * std::log(std::cosh(x));
    const double expected_grad = scale * std::tanh(x);
    const double expected_curvature =
        1.0 - std::tanh(x) * std::tanh(x);
    require_close(ForceResidualLoss(residual, ForceLossKind::LogCosh, scale),
                  expected_loss, 1.0e-13, "log-cosh loss");
    require_close(ForceResidualGrad(residual, ForceLossKind::LogCosh, scale),
                  expected_grad, 1.0e-13, "log-cosh grad");
    require_close(ForceResidualCurvature(
                      residual, ForceLossKind::LogCosh, scale),
                  expected_curvature, 1.0e-13, "log-cosh curvature");
    require_close(ForceResidualGrad(100.0, ForceLossKind::LogCosh, scale),
                  scale, 1.0e-12, "log-cosh positive saturation");
    require_close(ForceResidualGrad(-100.0, ForceLossKind::LogCosh, scale),
                  -scale, 1.0e-12, "log-cosh negative saturation");

    if (ParseForceLossKind("l2") != ForceLossKind::L2 ||
        ParseForceLossKind("log-cosh") != ForceLossKind::LogCosh ||
        std::string(ForceLossKindName(ForceLossKind::LogCosh)) != "log-cosh") {
        std::cerr << "force loss parser/name mismatch" << std::endl;
        return 1;
    }

    return 0;
}
CPP

g++ -std=c++11 -I"$ROOT" "$TMPDIR/check_force_loss.cpp" -o "$TMPDIR/check_force_loss"
"$TMPDIR/check_force_loss"

grep -q -- "--force-loss=<l2|log-cosh>" "$ROOT/src/mlp/mlp_commands.cpp"
grep -q -- "--force-log-cosh-scale=<double>" "$ROOT/src/mlp/mlp_commands.cpp"

echo "force_log_cosh_loss_check passed"
