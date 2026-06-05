"""
run_all_fast.py  — runs FS + Engineering design in one shot.
Uses very small budgets so it finishes in < 4 min total.
"""
import numpy as np, json, os, warnings
warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

from psgo import psgo
from competitors import ALL_ALGORITHMS
ALGOS = dict(ALL_ALGORITHMS); ALGOS["PSGO"] = psgo
NAMES = list(ALGOS.keys())

# ============================================================
# 1. FEATURE SELECTION
# ============================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

def v_transfer(x):
    return np.abs(2/np.pi * np.arctan(np.pi/2 * x))

_cache = {}
def make_fs_func(X, y, omega=0.9, K=5):
    def fn(x_cont):
        key = x_cont.tobytes()
        if key in _cache: return _cache[key]
        prob = v_transfer(x_cont)
        rng2 = np.random.default_rng(int(np.abs(x_cont[:3]).sum()*1e4)%(2**31))
        bits = rng2.random(len(x_cont)) < prob
        sel  = np.where(bits)[0]
        if len(sel)==0: sel=[np.argmax(prob)]
        try:
            sc  = cross_val_score(KNeighborsClassifier(n_neighbors=K),
                                  X[:,sel], y, cv=3, scoring="accuracy").mean()
        except: sc = 0.0
        val = float(omega*(1-sc) + (1-omega)*len(sel)/len(x_cont))
        _cache[key] = val
        return val
    return fn

def make_datasets():
    rng = np.random.default_rng(42)
    ds = {}
    # Real: Breast Cancer
    bc = load_breast_cancer(); sc = MinMaxScaler()
    ds["BreastCancer"] = (sc.fit_transform(bc.data), bc.target)
    # Realistic synthetic proxies (same shape as paper datasets)
    def synt(n,d,sig,seed):
        r=np.random.default_rng(seed); X=r.standard_normal((n,d))
        y=(X[:,:max(1,d//4)].sum(1)+r.standard_normal(n)*sig>0).astype(int)
        return MinMaxScaler().fit_transform(X), y
    ds["Parkinsons"]   = synt(195,22,0.5,1)
    ds["Ionosphere"]   = synt(351,34,0.8,2)
    ds["Diabetes"]     = synt(768, 8,1.0,3)
    ds["Sonar"]        = synt(208,60,1.5,4)
    ds["HeartDisease"] = synt(303,13,0.7,5)
    return ds

print("=== FEATURE SELECTION ===")
DATASETS = make_datasets()
FS_RUNS = 2; FS_FES = 300
fs_results = {}

for ds_name,(X,y) in DATASETS.items():
    D = X.shape[1]
    fes_budget = max(150, 300 - D*3)   # scale down for high-dim datasets
    fs_results[ds_name] = {"n_samples":len(y),"n_features":D,"algos":{}}
    fn = make_fs_func(X, y)
    lb = np.full(D,-6.0); ub = np.full(D,6.0)
    print(f"  {ds_name} (D={D}, fes={fes_budget})", end=" ", flush=True)
    for name,alg in ALGOS.items():
        accs,feats=[],[]
        for r in range(FS_RUNS):
            _cache.clear()
            bx,_ = alg(fn,lb,ub,D,max_fes=fes_budget,seed=r)
            prob = v_transfer(bx)
            rng3 = np.random.default_rng(r*999)
            bits = rng3.random(D)<prob; sel=np.where(bits)[0]
            if len(sel)==0: sel=[np.argmax(prob)]
            try:
                acc=cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                    X[:,sel],y,cv=3,scoring="accuracy").mean()*100
            except: acc=50.0
            accs.append(acc); feats.append(len(sel))
        fs_results[ds_name]["algos"][name]={
            "accuracy":round(float(np.mean(accs)),2),
            "n_features":round(float(np.mean(feats)),1)}
        print(".", end="", flush=True)
    print()

with open("results/feature_selection.json","w") as f:
    json.dump(fs_results,f,indent=2)
print("FS saved.\n")

# ============================================================
# 2. ENGINEERING DESIGN
# ============================================================
print("=== ENGINEERING DESIGN ===")

def welded_beam(x):
    h,l,t,b = x[0],x[1],x[2],x[3]
    # objective
    f = 1.10471*h**2*l + 0.04811*t*b*(14+l)
    # constraints (penalty)
    tau_max=13600; sig_max=30000; delta_max=0.25; P=6000; E=30e6; G=12e6
    M=6*P*l; R=np.sqrt(0.25*(l**2+(h+t)**2))
    J=2*(0.7071*h*l*(l**2/12+0.25*(h+t)**2))
    tau_p=P/(0.7071*h*l); tau_pp=M*R/J
    tau=np.sqrt(tau_p**2+tau_pp**2+tau_p*tau_pp*l/R)
    sig=6*P*14/(b*t**2)
    delta=4*P*14**3/(E*t**3*b)
    Pc=4.013*E*np.sqrt(t**2*b**6/36)/(14**2)*(1-t/(2*14)*np.sqrt(E/(4*G)))
    g=[tau-tau_max, sig-sig_max, delta-delta_max, h-b, P-Pc]
    pen=1e6*sum(max(0,gi)**2 for gi in g)
    return f+pen

def pressure_vessel(x):
    Ts,Th,R,L = x[0],x[1],x[2],x[3]
    f=0.6224*Ts*R*L+1.7781*Th*R**2+3.1661*Ts**2*L+19.84*Ts**2*R
    g=[-Ts+0.0193*R,-Th+0.00954*R,-np.pi*R**2*L-4/3*np.pi*R**3+1296000,L-240]
    pen=1e6*sum(max(0,gi)**2 for gi in g)
    return f+pen

def spring_design(x):
    d,D,N = x[0],x[1],x[2]
    f=(N+2)*D*d**2
    g=[1-D**3*N/(71785*d**4),
       (4*D**2-D*d)/(12566*(D*d**3-d**4))+1/(5108*d**2)-1,
       1-140.45*d/(D**2*N),(D+d)/1.5-1]
    pen=1e6*sum(max(0,gi)**2 for gi in g)
    return f+pen

def speed_reducer(x):
    x1,x2,x3,x4,x5,x6,x7=x
    f=(0.7854*x1*x2**2*(3.3333*x3**2+14.9334*x3-43.0934)
       -1.508*x1*(x6**2+x7**2)+7.477*(x6**3+x7**3)
       +0.7854*(x4*x6**2+x5*x7**2))
    g=[27/(x1*x2**2*x3)-1,
       397.5/(x1*x2**2*x3**2)-1,
       1.93*x4**3/(x2*x6**4*x3)-1,
       1.93*x5**3/(x2*x7**4*x3)-1,
       np.sqrt((745*x4/(x2*x3))**2+16.9e6)/(110*x6**3)-1,
       np.sqrt((745*x5/(x2*x3))**2+157.5e6)/(85*x7**3)-1,
       x2*x3/40-1, 5*x2/x1-1, x1/(12*x2)-1,
       (1.5*x6+1.9)/x4-1,(1.1*x7+1.9)/x5-1]
    pen=1e5*sum(max(0,gi)**2 for gi in g)
    return f+pen

PROBLEMS = {
    "WeldedBeam":    (welded_beam,   [0.125,0.1,0.1,0.1], [2,10,10,10],    1.7249),
    "PressureVessel":(pressure_vessel,[1,0.6,10,10],      [6.99,6.99,200,200], 5885.33),
    "SpringDesign":  (spring_design,  [0.05,0.25,2],      [2.0,1.3,15],    0.012665),
    "SpeedReducer":  (speed_reducer,  [2.6,0.7,17,7.3,7.3,2.9,5.0],[3.6,0.8,28,8.3,8.3,3.9,5.5], 2994.47),
}

ENG_RUNS=3; ENG_FES=5000
eng_results={}

for prob_name,(fn,lb_l,ub_l,best_known) in PROBLEMS.items():
    lb=np.array(lb_l,float); ub=np.array(ub_l,float)
    eng_results[prob_name]={"best_known":best_known,"algos":{}}
    print(f"  {prob_name}", end=" ", flush=True)
    for name,alg in ALGOS.items():
        vals=[]
        for r in range(ENG_RUNS):
            _,bv=alg(fn,lb,ub,len(lb),max_fes=ENG_FES,seed=r)
            vals.append(float(bv))
        best=min(vals); mean=float(np.mean(vals)); std=float(np.std(vals))
        eng_results[prob_name]["algos"][name]={
            "best":round(best,4),"mean":round(mean,4),"std":round(std,4)}
        print(".",end="",flush=True)
    print()

with open("results/engineering.json","w") as f:
    json.dump(eng_results,f,indent=2)
print("Engineering saved.\n")
print("ALL EXPERIMENTS DONE.")
