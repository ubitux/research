"""
Find the best rotation coefficient approximations using integers
See also https://www.shadertoy.com/view/XlsyWX
"""

from math import cos, pi, sin, tau

N = 1000  # number of angles to test


def best(f, g, k: int) -> int:
    b = 1e38
    C = -1
    turn = 0
    while True:
        c = round(tau * turn + pi / 2 * k)  # rounded constant
        if c > 99:  # maximum significant digit
            break
        e = 0  # total accumulated error
        for i in range(N):
            a = i / N * tau  # test angle
            r, v = f(a), g(a + c)  # ref, approx
            e += abs(r - v)
        if e < b:
            b, C = e, c
        turn += 1
    return C


t0 = best(lambda a: sin(a), cos, 3)
t1 = best(lambda a: sin(-a), cos, 1)

# Equivalent to mat2(cos(a),sin(a),-sin(a),cos(a))
print(f"mat2(cos(a+vec4(0,{t0},{t1},0)))")
