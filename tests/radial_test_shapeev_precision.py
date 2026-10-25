"""90-digit, read-only decomposition of native Shapeev rounding sensitivity."""
import ast
from decimal import Decimal as D, getcontext
import json
from pathlib import Path
import re
import subprocess
import sys

getcontext().prec = 90


def evaluate(node):
    if isinstance(node, ast.Constant):
        return D(node.value) if isinstance(node.value, int) else D.from_float(node.value)
    if isinstance(node, ast.Name):
        return {'min_dist': D('.5'), 'max_dist': D(5)}[node.id]
    if isinstance(node, ast.UnaryOp):
        a = evaluate(node.operand)
        return -a if isinstance(node.op, ast.USub) else a
    if isinstance(node, ast.BinOp):
        a, b = evaluate(node.left), evaluate(node.right)
        return {ast.Add: lambda: a+b, ast.Sub: lambda: a-b,
                ast.Mult: lambda: a*b, ast.Div: lambda: a/b}[type(node.op)]()
    if isinstance(node, ast.Call):
        args = [evaluate(a) for a in node.args]
        if node.func.id == 'sqrt':
            return args[0].sqrt()
        if node.func.id == 'pow':
            return args[0] ** int(args[1])
    raise ValueError(ast.dump(node))


header = Path(__file__).resolve().parents[1] / 'interfaces/lammps/ML-SUS2-KK/sus2_sl_kokkos_radial.h'
expressions = re.findall(r'rb_coeffs\[(\d+)\]\[(\d+)\]\s*=\s*([^;]+);', header.read_text())
ideal = {(int(n), int(i)-2): evaluate(ast.parse(expr, mode='eval').body)
         for n, i, expr in expressions}
raw = subprocess.check_output([sys.argv[1]], text=True)
coeff, values, mismatches = {}, {}, 0
for line in raw.splitlines():
    row = line.split()
    if row[0] == 'COEFF':
        a, b = float.fromhex(row[3]), float.fromhex(row[4])
        coeff[int(row[1]), int(row[2])] = a
        mismatches += a.hex() != b.hex()
    elif row[0] == 'VALUE':
        values[int(row[1])] = list(map(float, row[2:]))
r, scaling = D('2.5'), D.from_float(.73)
result = {'precision_digits': 90, 'cpu_device_coefficient_bit_mismatches': mismatches, 'rows': []}
for n in (9, 10, 11):
    exact = sum((scaling*D.from_float(coeff[n, i])*r**i for i in range(-2, n+1)), D(0))
    scaled_exact = sum((D.from_float(.73*coeff[n, i])*r**i for i in range(-2, n+1)), D(0))
    ideal_value = sum((scaling*ideal[n, i]*r**i for i in range(-2, n+1)), D(0))
    abs_sum = sum((abs(scaling*D.from_float(coeff[n, i])*r**i) for i in range(-2, n+1)), D(0))
    cpu, plain, fma, integer = values[n]
    result['rows'].append({'n': n, 'r': str(r), 'cpu': cpu, 'plain': plain, 'fma': fma,
                          'integer_power_fma': integer, 'exact_stored_coeff': str(exact),
                          'exact_rounded_scaled_coeff': str(scaled_exact),
                          'ideal_formula_coeff_value': str(ideal_value),
                          'sum_abs_terms': str(abs_sum), 'condition_sum_abs_over_value': str(abs_sum/abs(exact)),
                          'cpu_minus_exact': str(D.from_float(cpu)-exact),
                          'fma_minus_exact': str(D.from_float(fma)-exact),
                          'fma_minus_cpu': str(D.from_float(fma)-D.from_float(cpu))})
print(json.dumps(result, indent=2))
