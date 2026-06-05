"""
PSGO v4: Pistol Shrimp-Goby Optimization — Enhanced
=====================================================
Improvements (all validated to actually help):

1. OBL Initialization — but ONLY for G (Gobies), not S
   Minimal extra cost (N evals), better starting explorer positions

2. Best-guided differential in Goby phase
   G_new uses x_star more aggressively via random weight

3. Shrimp center: weighted toward x_star (0.7 x_star + 0.3 G[i])
   Tighter exploitation around best

4. Stagnation recovery: if no improvement for T_max/5 iterations,
   reinit worst 20% of Gobies (Eq-ext: partial restart)

All changes: N=30, T_blast=50 unchanged (proven safe values)
"""
import numpy as np
from math import gamma, pi


def levy_flight(lam, D, rng):
    sigma_u = (gamma(1+lam)*np.sin(pi*lam/2)
               / (gamma((1+lam)/2)*lam*2**((lam-1)/2)))**(1/lam)
    u = rng.normal(0, sigma_u, D)
    v = rng.normal(0, 1, D)
    return u / (np.abs(v)**(1/lam))


def psgo(func, lb, ub, dim, max_fes,
         N=30, alpha=0.5, beta=0.3, lam=1.5, sigma2=0.1,
         r0=0.5, eta=2.0, theta_s=0.3, theta_h=0.7,
         T_blast=50, gamma_pull=0.5, k_max=20,
         seed=None, record_convergence=False):

    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    span = ub - lb
    T_max = max(1, max_fes // (2*N))
    fes   = 0
    conv  = []
    stag  = 0               # stagnation counter
    T_stag = max(10, T_max // 5)  # reinit trigger

    def ev(x):
        nonlocal fes; fes += 1; return func(x)

    # ---- Phase 0: Init ----
    # G (Gobies): OBL — pick better of random vs opposite
    G_rand = lb + rng.random((N, dim)) * span
    G_opp  = np.clip(lb + ub - G_rand, lb, ub)
    fGr    = np.array([ev(G_rand[i]) for i in range(N)])
    fGo    = np.array([ev(G_opp[i])  for i in range(N)])
    G      = np.where((fGr <= fGo)[:,None], G_rand, G_opp)
    fG     = np.minimum(fGr, fGo)

    # S (Shrimps): standard init (save evals for main loop)
    S  = lb + rng.random((N, dim)) * span
    fS = np.array([ev(S[i]) for i in range(N)])

    k = np.zeros(N, dtype=int)
    allf = np.concatenate([fG, fS]); allx = np.vstack([G, S])
    b = np.argmin(allf)
    x_star = allx[b].copy(); f_star = allf[b]
    f_prev = f_star

    t = 0
    while fes < max_fes:
        t += 1
        decay = max(1.0 - t/T_max, 0.0)
        a_t   = alpha*(0.1 + 0.9*decay)

        for i in range(N):
            # ---- Phase 1: Explorer (Goby) — Levy (Eq3) ----
            ell    = levy_flight(lam, dim, rng)
            r1, r2 = rng.integers(N), rng.integers(N)
            diff   = G[r1] - G[r2]
            # Random weight on x_star pull (more aggressive when r<0.5)
            w_star = 0.5 + 0.5*rng.random()
            G_new  = G[i] + a_t*ell*(x_star - G[i])*w_star + a_t*0.5*diff
            G_new  = np.clip(G_new, lb, ub)
            fGn    = ev(G_new)
            if fGn < fG[i]:
                G[i] = G_new; fG[i] = fGn
            if fG[i] < f_star:
                x_star = G[i].copy(); f_star = fG[i]

            # ---- Phase 2: Exploiter (Shrimp) — (Eq5) ----
            dist   = np.abs(x_star - G[i]) + 1e-12
            ls     = beta * np.sqrt(sigma2) * (0.02 + 0.98*decay)
            # Weighted center: 0.7 toward x_star, 0.3 toward G[i]
            center = 0.7*x_star + 0.3*G[i]
            S_base = center + rng.normal(0, 1, dim)*ls*dist

            # ---- Phase 3: GTDS danger + reset (Eq6,7) ----
            delta_f = abs(fS[i] - f_star)
            f_rng   = (fS.max() - fS.min()) + 1e-10
            delta   = (k[i]/k_max)*np.exp(-delta_f/f_rng)

            if delta > theta_h:
                S_cand = lb + rng.random(dim)*span
                k[i]   = 0
            elif delta > theta_s:
                eps    = rng.normal(0, 1, dim)*0.001*dist
                S_cand = gamma_pull*S_base + (1-gamma_pull)*x_star + eps
            else:
                S_cand = S_base

            S_cand = np.clip(S_cand, lb, ub)
            fSc    = ev(S_cand)
            if fSc < fS[i]:
                S[i] = S_cand; fS[i] = fSc; k[i] = 0
            else:
                k[i] = min(k[i]+1, k_max)
            if fS[i] < f_star:
                x_star = S[i].copy(); f_star = fS[i]

            if fes >= max_fes:
                break

        # ---- Phase 4: Claw-Blast (Eq8,9) ----
        if t % T_blast == 0 and fes < max_fes:
            r_t = r0*np.exp(-eta*t/T_max)
            xb  = np.clip(x_star + r_t*rng.uniform(-1,1,dim)*span, lb, ub)
            fb  = ev(xb)
            if fb < f_star:
                x_star = xb.copy(); f_star = fb

        # ---- Phase 5: Stagnation recovery (Eq-ext) ----
        if f_star < f_prev:
            stag   = 0
            f_prev = f_star
        else:
            stag += 1

        if stag >= T_stag and fes < max_fes - N:
            # Reinit worst 20% of Gobies randomly
            n_reinit = max(1, N // 5)
            worst_G  = np.argsort(fG)[-n_reinit:]
            for wi in worst_G:
                G[wi]  = lb + rng.random(dim)*span
                fG[wi] = ev(G[wi])
                if fG[wi] < f_star:
                    x_star = G[wi].copy(); f_star = fG[wi]
            stag = 0

        if record_convergence:
            conv.append((fes, f_star))

    if record_convergence:
        return x_star, f_star, conv
    return x_star, f_star


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    def sphere(x): return np.sum(x**2)
    def rastrigin(x): return 10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x))
    def rosenbrock(x): return np.sum(100*(x[1:]-x[:-1]**2)**2+(x[:-1]-1)**2)
    def ackley(x):
        n=len(x)
        return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))-np.exp(np.sum(np.cos(2*np.pi*x))/n)+20+np.e
    D=30; FES=30000
    print("PSGO v4 Test"); print("="*40)
    for fn,name,l,u in [(sphere,"Sphere",-100,100),(rosenbrock,"Rosenbrock",-30,30),
                         (rastrigin,"Rastrigin",-5.12,5.12),(ackley,"Ackley",-32,32)]:
        lb=np.full(D,float(l)); ub=np.full(D,float(u))
        res=[psgo(fn,lb,ub,D,max_fes=FES,seed=s)[1] for s in range(10)]
        print(f"  {name:12s}: {np.mean(res):.4e}")
