import numpy as np
import sympy as sp
import math

r = (-1 + math.sqrt(5))/2
fibo = [0, 1, 1,]
for i in range(3, 1000):
    fibo.append(fibo[i-1] + fibo[i-2])


def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        print("정규화 : ", v)
    else:
        print("정규화 : ", v / norm)


def pca_covariance(points):
    m = np.array([points[0]], np.float64)
    tmp = [np.array(points[0], np.float64), ]
    first = True
    for p in points:
        if first:
            first = False
            continue
        tmp.append(np.array([p], np.float64))
        m += tmp[-1]
    m /= len(points)
    c = (tmp[0] - m) * (tmp[0] - m).T
    first = True
    for p in tmp:
        if first:
            first = False
            continue
        c += (p-m) * (p-m).T
    c /= len(points)
    print("Avg m : \n", m)
    print("Covariance C : \n", c)


def lu_decomposition(m):
    l, u = np.zeros((len(m), len(m))), np.zeros((len(m), len(m)))
    for i in range(len(m)):
        for j in range(len(m)):
            if i == 0:
                u[0][j] = m[0][j]
            if j == 0:
                l[i][0] = m[i][0]/u[0][0]
            for o in range(2):
                if j > 0:
                    l[i][j] = m[i][j]
                    for k in range(j):
                        l[i][j] -= l[i][k] * u[k][j]
                    if u[j][j] != 0:
                        l[i][j] /= u[j][j]
                    else:
                        l[i][j] = 0
                if i > 0:
                    u[i][j] = m[i][j]
                    for k in range(i):
                        u[i][j] -= l[i][k]*u[k][j]

    print("L :\n", l)
    print("U :\n", u)


def convergence_check(A):
    A = np.array(A)
    sumA = np.sum(A, axis=1)
    for i in range(len(A)):
        if 2*abs(A[i][i]) > abs(sumA[i]):
            continue
        return False
    return True


def jacobi(A, B, P, n):
    if convergence_check(A):
        print("Always converge!")
    A = np.array(A)
    B = np.array(B)
    print("A :\n", A, "\nB : \n", B)
    print("0 번: x:", P[0], "y:", P[1], "z:", P[2])
    for i in range(n):
        x = (B[0] - A[0][1] * P[1] - A[0][2] * P[2]) / A[0][0]
        y = (B[1] - A[1][0] * P[0] - A[1][2] * P[2]) / A[1][1]
        z = (B[2] - A[2][0] * P[0] - A[2][1] * P[1]) / A[2][2]
        print(i+1, "번: x:", x, "y:", y, "z:", z)
        P[0], P[1], P[2] = x, y, z


def gauss_seidel(A, B, P, n, relax=1.0):
    if convergence_check(A):
        print("Always converge!")
    A = np.array(A)
    B = np.array(B)
    print("A :\n", A, "\nB : \n", B)
    print("0 번: x:", P[0], "y:", P[1], "z:", P[2])
    for i in range(n):
        x = relax * ((B[0] - A[0][1] * P[1] - A[0][2] * P[2]) / A[0][0]) + ((1-relax) * P[0])
        y = relax * ((B[1] - A[1][0] * x - A[1][2] * P[2]) / A[1][1]) + ((1-relax) * P[1])
        z = relax * ((B[2] - A[2][0] * x - A[2][1] * y) / A[2][2]) + ((1-relax) * P[2])
        print(i+1, "번: x:", x, "y:", y, "z:", z)
        P[0], P[1], P[2] = x, y, z


def interpolation(points, target, n):
    L = []
    P = 0
    for k in range(n):
        up = 1
        down = 1
        for pointIdx in range(len(points)):
            if pointIdx == k:
                continue
            up *= target - points[pointIdx][0]
            down *= points[k][0] - points[pointIdx][0]
        L.append(up / down)
        print("L", k, ":", up/down)
    for i in range(n):
        P += points[i][1]*L[i]
    print("P", n-1, "(", target, "):", P)


def bilinear_interpol(X, Y, F, P):
    F_R1 = (X[1] - P[0])/(X[1] - X[0])*F[0] + (P[0] - X[0])/(X[1] - X[0])*F[1]
    F_R2 = (X[1] - P[0])/(X[1] - X[0])*F[2] + (P[0] - X[0])/(X[1] - X[0])*F[3]
    F_P = (Y[1] - P[1])/(Y[1] - Y[0])*F_R1 + (P[1] - Y[0])/(Y[1] - Y[0])*F_R2
    print("bilinear:", F_P)
    return F_P


def trilinear_interpol(X, Y, Z, F, P):
    F_D = bilinear_interpol(X, Y, F[0:4], P[0:2])
    F_U = bilinear_interpol(X, Y, F[4:8], P[0:2])
    F_P = (Z[1] - P[2])/(Z[1] - Z[0])*F_D + (P[2] - Z[0])/(Z[1] - Z[0])*F_U
    print("trilinear", F_P)


def least_squares(form, points, target):
    F = np.zeros((len(points), len(form)))
    Y = np.array(points)[:, 1]
    for i in range(len(points)):
        for j in range(len(form)):
            F[i, j] = form[j](points[i][0])
    ans = np.linalg.solve(F.T @ F, F.T @ Y.T)
    print("form:", ans)
    value = 0
    for i in range(len(form)):
        value += form[i](target) * ans[i]
    print("value:", value)


def data_linearization(points, value=1):
    x, y, x2, xy = 0, 0, 0, 0
    for point in points:
        x += point[0]
        y += np.log(point[1])
        x2 += point[0] ** 2
        xy += np.log(point[1])*point[0]
    A = np.array([[x2, x], [x, len(points)]])
    B = np.array([[xy], [y]])
    C = np.linalg.solve(A, B)
    print("Y =", C[0][0], "X +", C[1][0])
    print("C =", math.exp(C[1][0]))
    print("y =", math.exp(C[1][0]), "e^", C[0][0], "x")
    print("f(", value, "):", math.exp(C[1][0]) * math.exp(C[0][0] * value))


def golden_ratio_search(form, a, b, cnt):
    global r
    if cnt == 0:
        return
    c = a + (1 - r) * (b-a)
    d = b - (1 - r) * (b-a)
    fc, fd = 0, 0
    for i in form:
        fc += i(c)
        fd += i(d)
    print(cnt, "a:", a, "c:", c, "d:", d, "b:", b, "f(c):", fc, "f(d):", fd)
    if fc > fd:
        golden_ratio_search(form, c, b, cnt-1)
    else:
        golden_ratio_search(form, a, d, cnt-1)


def find_fibo_n(a, b, tolerance):
    i = 0
    while fibo[i] <= (b-a)/tolerance:
        i += 1
    return i


def fibonacci_search(form, a, b, n, e):
    if n <= 2:
        print('if f(c) > f(d), update di = ci + ' + str(e) + '. if f(c) < f(d), update ci = di - ' + str(e))
        return
    c = a + (1 - fibo[n-1]/fibo[n]) * (b-a)
    d = a + fibo[n-1]/fibo[n] * (b-a)
    fc, fd = 0, 0
    for i in form:
        fc += i(c)
        fd += i(d)
    print('n-' + str(n) + ' 번째')
    print("a:", a, "\nc:", c, "\nd:", d, "\nb:", b)
    if fc > fd:
        print('f(c) > f(d). select c as new a value\n')
        fibonacci_search(form, c, b, n - 1, e)
    else:
        print('f(c) < f(d). select d as new b value\n')
        fibonacci_search(form, a, d, n - 1, e)


def hessian(f, target):
    diff_x2 = sp.diff(sp.diff(f, x), x).subs([(x, target[0]), (y, target[1])])
    diff_xy = sp.diff(sp.diff(f, y), x).subs([(x, target[0]), (y, target[1])])
    diff_y2 = sp.diff(sp.diff(f, y), y).subs([(x, target[0]), (y, target[1])])
    h = diff_x2 * diff_y2 - diff_xy ** 2
    print("|H|:", h)
    if h > 0:
        if diff_x2 > 0:
            print("f(", target, ") has a local minimum")
        elif diff_x2 < 0:
            print("f(", target, ") has a local maximum")
    elif h < 0:
        print("f(", target, ") has a saddle point")
    else:
        print("it's nothing")


def steepest_ascend(f, start, range):
    if range[0] == range[1]:
        return
    print(range[0], "번째= x:", start[0], "y:", start[1], end=' ')
    diff_x = sp.diff(f, x).subs([(x, start[0]), (y, start[1])])
    diff_y = sp.diff(f, y).subs([(x, start[0]), (y, start[1])])
    diff_x = float(diff_x)
    diff_y = float(diff_y)
    h = sp.symbols('h')
    G = f.subs([(x, start[0] + diff_x * h), (y, start[1] + diff_y * h)])
    g = sp.diff(G, h)
    h_value = sp.solvers.solve(g, h)[0]
    print("h:", h_value)
    steepest_ascend(f, [start[0] + diff_x*h_value, start[1] + diff_y * h_value], [range[0]+1, range[1]])


def bisection(f, start, range, error=0.001):
    if range[0] == range[1]:
        return
    print(range[0], "번째 l~u:", start)
    fl = f.subs([(x, start[0])])
    fu = f.subs([(x, start[1])])
    m = (start[0]+start[1]) / 2
    fm = f.subs([(x, m)])
    if fm*fl < 0:
        nm = (start[0] + m)/2
        print("e:", abs((nm - m)/nm)*100, end=' ')
        if abs((nm - m)/nm)*100 > error:
            bisection(f, [start[0], m], [range[0] + 1, range[1]])
    elif fm*fl > 0:
        nm = (start[1] + m)/2
        print("e:", abs((nm - m)/nm)*100, end=' ')
        if abs((nm - m)/nm)*100 > error:
            bisection(f, [m, start[1]], [range[0] + 1, range[1]])
    else:
        return


def newton_raphson(f, start, range, error=10 ** -8):
    if range[0] == range[1]:
        return
    print(range[0], "번째 x:", start)
    fd = sp.diff(f, x).subs([(x, start)])
    fd = float(fd)
    fx = f.subs([(x, start)])
    fx = float(fx)
    try:
        next = start - (fx / fd)
    except ZeroDivisionError:
        print("f'(x) is zero")
        return
    if abs((next - start)/next)*100 > error:
        newton_raphson(f, next, [range[0]+1, range[1]], error)


def secant(f, pre, now, range, error=10 ** -8):
    if range[0] == range[1]:
        return
    fnow = f.subs([(x, now)])
    fpre = f.subs([(x, pre)])
    next = now - (fnow*(now - pre)/(fnow - fpre))
    e = abs((next-now)/next) * 100
    print(range[0], "번째 x-1:", pre, "x:", now, "x+1:", next)
    if e > error:
        secant(f, now, next, [range[0]+1, range[1]], error)


def euler_method(f, init, target, h):
    xi, yi, zi = init
    print("x0: ", xi, " y0: ", yi, " y'0: ", zi)
    for i in range(target):
        tmp = zi
        zi = zi + h*f.subs([(x, xi), (y, yi), (z, zi)])
        zi = float(zi)
        xi = xi + h
        yi = yi + h*tmp
        print("x{0}: ".format(i+1), xi, " y{0}: ".format(i+1), yi, " y'{0}: ".format(i+1), zi)
    print("y({0})= {1}".format(xi, yi))


def modified_euler_method(f, init, target, h):
    xi, yi = init
    for i in range(5):
        print("x{0}: ".format(i), xi, " y{0}: ".format(i), yi)
        yi = yi + h * (f.subs([(x, xi + h/2), (y, yi + h/2*f.subs([(x, xi), (y, yi)]))]))
        yi = float(yi)
        xi = xi + h
        if xi == target:
            print("x{0}: ".format(i + 1), xi, " y{0}: ".format(i + 1), yi)
            print("y({0})= {1}".format(xi, yi))
            return


def heun_method(f, init, target, h):
    xi, yi = init
    for i in range(5):
        print("x{0}: ".format(i), xi, " y{0}: ".format(i), yi)
        yi = yi + (h/2)*(f.subs([(x, xi), (y, yi)]) + f.subs([(x, xi+h), (y, yi + h*f.subs([(x, xi), (y, yi)]))]))
        yi = float(yi)
        xi = xi + h
        if xi == target:
            print("x{0}: ".format(i+1), xi, " y{0}: ".format(i+1), yi)
            print("y({0})= {1}".format(xi, yi))
            return


def newton_polynomial(points, target):
    a0 = points[0][1]
    a1 = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    a2 = ((points[2][1] - points[1][1]) / (points[2][0] - points[1][0]) - a1) / (points[2][0] - points[0][0])
    a3 = (((points[3][1] - points[2][1]) / (points[3][0] - points[2][0]) - (points[2][1] - points[1][1]) / (points[2][0] - points[1][0])) / (points[3][0] - points[1][0]) - a2) / (points[3][0] - points[0][0])
    print("a0: {0} a1: {1} a2: {2} a3: {3}".format(a0, a1, a2, a3))
    p1 = a0 + a1*(target - points[0][0])
    p2 = p1 + a2*(target - points[0][0])*(target - points[1][0])
    p3 = p2 + a3*(target - points[0][0])*(target - points[1][0])*(target - points[2][0])
    print("p1: {0} p2: {1} p3: {2}".format(p1, p2, p3))


def arr(A):
    return np.array(A,dtype=float)


def newton_polynomial2(P):
    P = arr(P)
    table = np.zeros(shape=(len(P),len(P)))
    table[:,0] = P[:,1]
    for i in range(1, len(P)):
        for j in range(i, len(P)):
            table[j,i] = (table[j,i-1] - table[j-1,i-1]) / i
    prev = table[0,0]
    for i in range(1, len(P)):
        func = sp.core.sympify(table[i,i])
        for j in range(i):
            func *= (x - P[j,0])
        func += prev
        prev = func
        print('P%d(x) = %s' % (i, sp.simplify(func)))


def hessian_3by3(f, p):
    H = np.zeros(shape=(len(p), len(p)))
    Q = (x, y, z)
    D = dict()
    for i in range(len(p)):
        D[Q[i]] = p[i]
    for i in range(len(p)):
        for j in range(len(p)):
            H[i,j] = sp.diff(sp.diff(f, Q[i]), Q[j]).subs(D)
    det = np.linalg.det(H)
    print(H)
    print('|H| = %8.5f' % det)


x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')

'''
# 정규화
normalize([1.5, 1.4, 1])

# [point1, point2, point3 ... point n] calculate m and C
# 벡터 구하는 건 행렬계산기 이용
pca_covariance([[-1, -2, 1], [1, 0, 2], [2, -1, 3], [2, -1, 2]])

# [[row1], [row2], [row3]] .. L U
# Mx = r -> Ux = A -> solve LA = R -> solve UX = A
lu_decomposition([[6, 4, 2], [3, -2, -1], [3, 4, 1]])

# Ax = B, [Mat A], [Mat B], [point1, point2, point3], num, (relax default = 1)
jacobi([[4, -1, 1], [4, -8, 1], [-2, 1, 5]], [7, -21, 15], [1, 2, 2], 20)
gauss_seidel([[4, -1, 1], [4, -8, 1], [-2, 1, 5]], [7, -21, 15], [1, 2, 2], 20)
# [[x0, y0], [x1, y1] ... [xn, yn]], target x, using # of points
interpolation([[1, 1.5574], [1.1, 1.9648], [1.2, 2.5722], [1.3, 3.6021]], 1.15, 4)
# [x0, x1], [y0, y1], [f(x0,y0), f(x1,y0), f(x0,y1), f(x1,y1)], [p0, p1]
bilinear_interpol([0, 1], [0, 1], [20, 120, 180, 200], [0.2, 0.5])
# [x0, x1], [y0, y1], [(0,0,0) -> (1,0,0) -> (0,1,0) -> (1,1,0) -> (0,0,1) -> (1,0,1) -> (0,1,1) -> (1,1,1)], [p1, p2, p3]
trilinear_interpol([0, 1], [0, 1], [0, 1], [80, 180, 100, 300, 10, 50, 40, 50], [0.2, 0.5, 0.6])

# [form; Ax^2 + Bx + C x:x**2 x: x x:1], [point1, point2 .. ] target
least_squares([lambda x:x**2, lambda x:x, lambda x:1], [[-1, 1], [0, 1], [1, 3], [2, 19]], 0.5)

# [point1, point2, point3 ... ], (value default = 1)
data_linearization([[0, 1.5], [1, 2.5], [2, 3.5], [3, 5.0], [4, 7.5]], 1)

# [form], interval [a, b], (find fibo_n) cnt /distinguish ability constant e
golden_ratio_search([lambda x:x**2, lambda x:-math.sin(x)], 0, 1, 20)
fibonacci_search([lambda x:x**2, lambda x:-math.sin(x)], 0, 1, find_fibo_n(0, 1, 10 ** -4), 0.01)

# polynomial , [x, y] cal hessian
hessian(x*(y**2), [1, 1])
hessian(2*x * (y**2) + 3*y + 2*x**2, [5, -1])
hessian_3by3(3*x**2+y**2+z**2, [-4, -2, 1])

# polynomial , [x, y] [range]
steepest_ascend(2*x*y + 2*x - x**2 - 2*y**2, [-1, 1], [0, 10])
# f, [xl, xu] [range] , error
bisection((9.8*68.1)/x * (1 - (sp.E ** (-(x/68.1)*10)))-40, [12, 16], [0, 10])

# f, start x0, range, error default = 10**-8
newton_raphson(sp.E ** -x - x, 0, [0, 10])
newton_raphson(2*x**3 - 4*x**2 + 1, 2, [0, 10])
# f, x-1, x, range, (error default = 10**-8)
secant(sp.E ** -x - x, 0, 1.0, [0, 10])

# euler's method ex) (y'' = 5x - 3y -y') y' == z 로 삽입
# f, [x0, y0, y'0], target num, h
euler_method(4*x + 2*y - z, [0, 1, 0], 2, 1)
modified_euler_method(3*x-2, [1, 0], 3, 1)
# heun's method ex) (y' = 3x - 2)
# f, [x0, y0], target, h
heun_method(3*x-2, [1, 0], 3, 1)
# [(x0, y0), (x1, y1),. ...,] , target
newton_polynomial([(-1, 1), (0, 1), (1, 3), (2, 19)], 0.5)
newton_polynomial2([(-1, 1), (0, 1), (1, 3), (2, 19)])
'''