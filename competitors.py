"""
Competitor metaheuristics for fair PSGO comparison.
All use the same (func, lb, ub, dim, max_fes, seed) interface and MINIMIZE.
Implementations follow standard published pseudocode.
"""
import numpy as np


def pso(func, lb, ub, dim, max_fes, N=30, w=0.729, c1=1.494, c2=1.494, seed=None):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    vmax = 0.2*span
    V = rng.uniform(-1, 1, (N, dim))*vmax
    fX = np.array([ev(X[i]) for i in range(N)])
    pbest = X.copy(); fpbest = fX.copy()
    g = np.argmin(fpbest); gbest = pbest[g].copy(); fgbest = fpbest[g]
    while fes < max_fes:
        for i in range(N):
            r1, r2 = rng.random(dim), rng.random(dim)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest-X[i])
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = np.clip(X[i]+V[i], lb, ub)
            f = ev(X[i])
            if f < fpbest[i]:
                pbest[i] = X[i].copy(); fpbest[i] = f
                if f < fgbest:
                    gbest = X[i].copy(); fgbest = f
            if fes >= max_fes: break
    return gbest, fgbest


def gwo(func, lb, ub, dim, max_fes, N=30, seed=None):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        idx = np.argsort(fX)
        Xa, Xb, Xd = X[idx[0]].copy(), X[idx[1]].copy(), X[idx[2]].copy()
        a = 2 - 2*t/T_max
        for i in range(N):
            new = np.zeros(dim)
            for lead in (Xa, Xb, Xd):
                A = 2*a*rng.random(dim)-a; C = 2*rng.random(dim)
                D = np.abs(C*lead - X[i])
                new += lead - A*D
            X[i] = np.clip(new/3, lb, ub)
            fX[i] = ev(X[i])
            if fes >= max_fes: break
    bi = np.argmin(fX)
    return X[bi].copy(), fX[bi]


def woa(func, lb, ub, dim, max_fes, N=30, b=1.0, seed=None):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    g = np.argmin(fX); best = X[g].copy(); fbest = fX[g]
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1; a = 2 - 2*t/T_max
        for i in range(N):
            r = rng.random(); A = 2*a*r - a; C = 2*rng.random()
            p = rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C*best - X[i]); X[i] = best - A*D
                else:
                    rand = X[rng.integers(N)]
                    D = np.abs(C*rand - X[i]); X[i] = rand - A*D
            else:
                D = np.abs(best - X[i]); l = rng.uniform(-1, 1, dim)
                X[i] = D*np.exp(b*l)*np.cos(2*np.pi*l) + best
            X[i] = np.clip(X[i], lb, ub)
            fX[i] = ev(X[i])
            if fX[i] < fbest:
                best = X[i].copy(); fbest = fX[i]
            if fes >= max_fes: break
    return best, fbest


def sca(func, lb, ub, dim, max_fes, N=30, a=2.0, seed=None):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    g = np.argmin(fX); best = X[g].copy(); fbest = fX[g]
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1; r1 = a - t*a/T_max
        for i in range(N):
            r2 = 2*np.pi*rng.random(dim); r3 = 2*rng.random(dim); r4 = rng.random(dim)
            step = np.where(r4 < 0.5, np.sin(r2), np.cos(r2))
            X[i] = X[i] + r1*step*np.abs(r3*best - X[i])
            X[i] = np.clip(X[i], lb, ub)
            fX[i] = ev(X[i])
            if fX[i] < fbest:
                best = X[i].copy(); fbest = fX[i]
            if fes >= max_fes: break
    return best, fbest


# ============================================================
#  Recent algorithms: HHO, AO, AVOA, ARO, INFO, SGA
# ============================================================

def hho(func, lb, ub, dim, max_fes, N=30, seed=None):
    """Harris Hawks Optimization."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    g = np.argmin(fX); rabbit = X[g].copy(); frabbit = fX[g]
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        E1 = 2*(1 - t/T_max)
        for i in range(N):
            E0 = 2*rng.random() - 1; E = E1*E0
            if abs(E) >= 1:  # exploration
                q = rng.random(); rand = X[rng.integers(N)]
                if q >= 0.5:
                    X[i] = rand - rng.random()*np.abs(rand - 2*rng.random()*X[i])
                else:
                    X[i] = (rabbit - X.mean(0)) - rng.random()*(lb + rng.random()*span)
            else:  # exploitation
                r = rng.random(); J = 2*(1 - rng.random())
                if r >= 0.5 and abs(E) < 0.5:
                    X[i] = rabbit - E*np.abs(rabbit - X[i])
                elif r >= 0.5 and abs(E) >= 0.5:
                    X[i] = (rabbit - X[i]) - E*np.abs(J*rabbit - X[i])
                else:
                    Y = rabbit - E*np.abs(J*rabbit - X[i])
                    X[i] = np.clip(Y, lb, ub)
            X[i] = np.clip(X[i], lb, ub)
            fX[i] = ev(X[i])
            if fX[i] < frabbit:
                rabbit = X[i].copy(); frabbit = fX[i]
            if fes >= max_fes: break
    return rabbit, frabbit


def ao(func, lb, ub, dim, max_fes, N=30, seed=None):
    """Aquila Optimizer."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    g = np.argmin(fX); best = X[g].copy(); fbest = fX[g]
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        Xmean = X.mean(0)
        for i in range(N):
            if t <= (2/3)*T_max:
                if rng.random() < 0.5:
                    X[i] = best*(1 - t/T_max) + (Xmean - best*rng.random())
                else:
                    rand = X[rng.integers(N)]
                    levy = _levy(dim, rng)
                    X[i] = best*levy + rand + (rng.random())*span*0.01
            else:
                if rng.random() < 0.5:
                    X[i] = (best - Xmean)*0.1 - rng.random() + (rng.random()*span)*0.01
                else:
                    QF = t**((2*rng.random()-1)/(1-T_max)**2)
                    G1 = 2*rng.random()-1; G2 = 2*(1 - t/T_max)
                    levy = _levy(dim, rng)
                    X[i] = QF*best - (G1*X[i]*rng.random()) - G2*levy + rng.random()*G1
            X[i] = np.clip(X[i], lb, ub)
            fX[i] = ev(X[i])
            if fX[i] < fbest:
                best = X[i].copy(); fbest = fX[i]
            if fes >= max_fes: break
    return best, fbest


def avoa(func, lb, ub, dim, max_fes, N=30, seed=None):
    """African Vultures Optimization Algorithm (simplified standard)."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        idx = np.argsort(fX)
        B1, B2 = X[idx[0]].copy(), X[idx[1]].copy()
        a = 2*rng.random()*(1 - t/T_max) - (1 - t/T_max)  # satiation
        for i in range(N):
            vulture = B1 if rng.random() < 0.5 else B2
            F = (2*rng.random()+1)*rng.random()*(1 - t/T_max) + a
            if abs(F) >= 1:  # exploration
                if rng.random() < 0.5:
                    X[i] = vulture - (np.abs(2*rng.random()*vulture - X[i]))*F
                else:
                    X[i] = vulture - F + rng.random()*((ub - lb)*rng.random() + lb)
            else:  # exploitation
                if rng.random() < 0.5:
                    d = vulture - X[i]
                    X[i] = vulture*(rng.random()*X[i]/(2*np.pi))*np.cos(X[i]) \
                           + vulture*(rng.random()*X[i]/(2*np.pi))*np.sin(X[i])
                    X[i] = vulture - np.abs(d)*F*_levy(dim, rng)
                else:
                    A1 = B1 - (B1*X[i])/(B1 - X[i]**2 + 1e-10)*F
                    A2 = B2 - (B2*X[i])/(B2 - X[i]**2 + 1e-10)*F
                    X[i] = (A1 + A2)/2
            X[i] = np.clip(X[i], lb, ub)
            fX[i] = ev(X[i])
            if fes >= max_fes: break
    bi = np.argmin(fX)
    return X[bi].copy(), fX[bi]


def aro(func, lb, ub, dim, max_fes, N=30, seed=None):
    """Artificial Rabbits Optimization."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    g = np.argmin(fX); best = X[g].copy(); fbest = fX[g]
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        A = 4*(1 - t/T_max)*np.log(1/rng.random())  # energy factor
        for i in range(N):
            if A > 1:  # detour foraging (exploration)
                j = rng.integers(N)
                while j == i: j = rng.integers(N)
                R = _levy(dim, rng)
                X[i] = X[j] + R*(X[i] - X[j]) + np.round(0.5*(0.05 + rng.random()))*rng.normal(0, 1, dim)
            else:  # random hiding (exploitation)
                r = rng.random(dim)
                X[i] = X[i] + A*((best - X[i])*rng.random() - r*X[i]*0.1)
            X[i] = np.clip(X[i], lb, ub)
            f = ev(X[i])
            if f < fX[i]: fX[i] = f
            if f < fbest:
                best = X[i].copy(); fbest = f
            if fes >= max_fes: break
    return best, fbest


def info(func, lb, ub, dim, max_fes, N=30, seed=None):
    """INFO: weighted mean of vectors (simplified)."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    X = lb + rng.random((N, dim))*span
    fX = np.array([ev(X[i]) for i in range(N)])
    T_max = max(1, max_fes // N); t = 0
    while fes < max_fes:
        t += 1
        idx = np.argsort(fX)
        xb, xb2, xw = X[idx[0]], X[idx[1]], X[idx[-1]]
        fb, fb2, fw = fX[idx[0]], fX[idx[1]], fX[idx[-1]]
        alpha = 2*np.exp(-4*t/T_max)
        for i in range(N):
            # weighted mean rule
            eps = 1e-25
            w1 = np.cos(fb - fX[i] + np.pi)*np.exp(-abs(fb - fX[i])/(fw - fb + eps))
            w2 = np.cos(fb2 - fX[i] + np.pi)*np.exp(-abs(fb2 - fX[i])/(fw - fb + eps))
            WM = alpha*(w1*(xb - X[i]) + w2*(xb2 - X[i])) / 3
            j = rng.integers(N)
            if rng.random() < 0.5:
                new = X[i] + WM + rng.normal(0, 1, dim)*(xb - X[j])*0.05
            else:
                new = xb + WM + rng.normal(0, 1, dim)*(xb - X[j])*0.05
            new = np.clip(new, lb, ub)
            f = ev(new)
            if f < fX[i]:
                X[i] = new; fX[i] = f
            if fes >= max_fes: break
    bi = np.argmin(fX)
    return X[bi].copy(), fX[bi]


def sga(func, lb, ub, dim, max_fes, N=30, a1=0.8, a2=0.5, beta_thr=0.05, seed=None):
    """Shrimp and Goby Association Search Algorithm (the direct precedent).
    Binary danger signal, homogeneous-ish update (no fixed roles, no claw-blast).
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    fes = 0
    def ev(x):
        nonlocal fes; fes += 1; return func(x)
    G = lb + rng.random((N, dim))*span   # goby
    S = lb + rng.random((N, dim))*span   # shrimp
    fG = np.array([ev(G[i]) for i in range(N)])
    fS = np.array([ev(S[i]) for i in range(N)])
    allf = np.concatenate([fG, fS]); allx = np.vstack([G, S])
    b = np.argmin(allf); best = allx[b].copy(); fbest = allf[b]
    T_max = max(1, max_fes // (2*N)); t = 0
    while fes < max_fes:
        t += 1
        for i in range(N):
            # goby update: move toward best
            G[i] = G[i] + a1*rng.normal(0, 1, dim)*(best - G[i])
            G[i] = np.clip(G[i], lb, ub)
            fG[i] = ev(G[i])
            if fG[i] < fbest: best = G[i].copy(); fbest = fG[i]
            # shrimp update near goby
            S[i] = G[i] + a2*rng.normal(0, 1, dim)*span*0.1*(1 - t/T_max)
            # binary danger signal
            if rng.random() < beta_thr:
                S[i] = best + rng.normal(0, 1, dim)*span*0.01  # full retreat
            S[i] = np.clip(S[i], lb, ub)
            fS[i] = ev(S[i])
            if fS[i] < fbest: best = S[i].copy(); fbest = fS[i]
            if fes >= max_fes: break
    return best, fbest


def _levy(D, rng, lam=1.5):
    from math import gamma as G, pi as P, sin
    sig = (G(1+lam)*sin(P*lam/2)/(G((1+lam)/2)*lam*2**((lam-1)/2)))**(1/lam)
    u = rng.normal(0, sig, D); v = rng.normal(0, 1, D)
    return u/(np.abs(v)**(1/lam))


# Registry of all competitors
ALL_ALGORITHMS = {
    'PSO': pso, 'GWO': gwo, 'WOA': woa, 'SCA': sca,
    'HHO': hho, 'AO': ao, 'AVOA': avoa, 'ARO': aro,
    'INFO': info, 'SGA': sga,
}
