import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel, GmlvqModel, LgmlvqModel
from mpl_toolkits.mplot3d import Axes3D

# === PARAMETRY GLOBALNE ===
prototypes_per_class = 5
initial_prototypes   = None      # None -> losowa inicjalizacja
max_iter              = 5000
gtol                  = 1e-5
beta                  = 2
random_state          = 42
display               = False
n_thresholds          = 200      # dla ROC

# === DANE ===
iris         = load_iris()
X            = iris.data
y            = iris.target
target_names = iris.target_names

# Skalowanie i podzia³
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=random_state
)

# --- Funkcje wizualizacji i metryk ---

def plot_prototypes(model, X, y, title):
    pca = PCA(2); Xp = pca.fit_transform(X)
    try:
        P = pca.transform(model.w_); labels = model.c_w_
    except AttributeError:
        print(f"{type(model).__name__}: brak prototypów.")
        return
    plt.figure(figsize=(7,5))
    plt.scatter(Xp[:,0], Xp[:,1], c=y, cmap='viridis', alpha=0.4)
    plt.scatter(P[:,0], P[:,1], c=labels, cmap='tab10',
                marker='X', s=150, edgecolor='k')
    plt.title(title); plt.grid(True); plt.show()

def plot_transformation_matrix(model, title):
    if hasattr(model,'omega_'):
        M = model.omega_; mat = M.T@M
    elif hasattr(model,'omegas_'):
        O = model.omegas_[0]; mat = O.T@O; title += " (1. prototyp)"
    else:
        print(f"{type(model).__name__}: brak transformacji."); return
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title); plt.xlabel("Cecha"); plt.ylabel("Cecha"); plt.show()

def plot_roc_auc_dense(model, X_test, y_test, name, n_thresh=100):
    W = model.w_
    if hasattr(model,'omega_'):
        Xp = X_test @ model.omega_.T; Wp = W @ model.omega_.T
    elif hasattr(model,'omegas_'):
        O = model.omegas_[0]; Xp = X_test @ O.T; Wp = W @ O.T
    else:
        Xp = X_test; Wp = W
    d = np.linalg.norm(Xp[:,None,:]-Wp[None,:,:], axis=2)
    inv = 1/(d+1e-12); proto_p = inv/inv.sum(1,keepdims=True)
    n_cls = len(target_names)
    probs = np.zeros((len(y_test), n_cls))
    for cls in range(n_cls):
        probs[:,cls] = proto_p[:, model.c_w_==cls].sum(1)
    fpr={}; tpr={}; aucv={}
    for i in range(n_cls):
        fpr[i], tpr[i] = roc_on_thresholds((y_test == i).astype(int), probs[:, i], n_thresh)
        idx = np.argsort(fpr[i]); aucv[i] = np.trapz(tpr[i][idx], fpr[i][idx])
    macro_auc = np.mean(list(aucv.values()))
    print(f"\nAUC-ROC (dense) {name}:")
    for i, cls in enumerate(target_names):
        print(f"  {cls}: {aucv[i]:.3f}")
    print(f"  Macro AUC: {macro_auc:.3f}")
    plt.figure(figsize=(7,5))
    for i,col in zip(range(n_cls), ['r','g','b']):
        plt.plot(fpr[i], tpr[i], col, lw=2, label=f"{target_names[i]} (AUC={aucv[i]:.2f})")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC – {name}"); plt.legend(loc='lower right'); plt.grid(True); plt.show()


def roc_on_thresholds(y_true_bin, scores, n_thresh=100):
    thr = np.linspace(0,1,n_thresh)
    tpr = np.zeros(n_thresh); fpr = np.zeros(n_thresh)
    P = y_true_bin.sum(); N = len(y_true_bin)-P
    for i, t in enumerate(thr):
        y_pred = (scores>=t).astype(int)
        tp = np.logical_and(y_pred==1, y_true_bin==1).sum()
        fp = np.logical_and(y_pred==1, y_true_bin==0).sum()
        tpr[i] = tp/P if P>0 else 0
        fpr[i] = fp/N if N>0 else 0
    return fpr, tpr

def train_and_evaluate(model, name):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    plot_prototypes(model, X_scaled, y, f"{name} – prototypy")
    plot_transformation_matrix(model, f"{name} – transformacja")
    plot_roc_auc_dense(model, X_test, y_test, name, n_thresholds)

# === Funkcja do rysowania wykresu 3D w zale¿noœci od parametrów ===
def plot_param_dependency_3d(model_class, param_grid, fixed_params, X, y, metric='accuracy', title="Wykres 3D"):
    Xp, Yp = np.meshgrid(param_grid['x'], param_grid['y'])
    Z = np.zeros_like(Xp, dtype=float)
    
    for i, x in enumerate(param_grid['x']):
        for j, y_val in enumerate(param_grid['y']):
            params = fixed_params.copy()
            params[param_grid['x_name']] = int(x)
            params[param_grid['y_name']] = int(y_val)
            model = model_class(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            if metric == 'accuracy':
                Z[j, i] = accuracy_score(y, y_pred)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xp, Yp, Z, cmap='viridis')
    ax.set_xlabel(param_grid['x_name'])
    ax.set_ylabel(param_grid['y_name'])
    ax.set_zlabel(metric)
    ax.set_title(title)
    plt.show()


# === INICJALIZACJA I EWALUACJA MODELI ===
glvq  = GlvqModel(prototypes_per_class, initial_prototypes,
                  max_iter=max_iter, gtol=gtol,
                  beta=beta, random_state=random_state,
                  display=display)
gmlvq = GmlvqModel(prototypes_per_class, initial_prototypes,
                   max_iter=max_iter, gtol=gtol,
                   beta=beta, random_state=random_state,
                   display=display)
lgmlvq= LgmlvqModel(prototypes_per_class, initial_prototypes,
                   max_iter=max_iter, gtol=gtol,
                   beta=beta, random_state=random_state,
                   display=display)

train_and_evaluate(glvq,  "GLVQ")
train_and_evaluate(gmlvq, "GMLVQ")
train_and_evaluate(lgmlvq,"LGMLVQ")

# === WYKRESY 3D dla wszystkich modeli: beta vs prototypes_per_class ===
models = {
    "GLVQ": GlvqModel,
    "GMLVQ": GmlvqModel,
    "LGMLVQ": LgmlvqModel
}

for name, model_class in models.items():
    print(f"\nTworzenie wykresu 3D dla {name}...")
    plot_param_dependency_3d(
        model_class,
        param_grid={
            'x_name': 'beta',
            'y_name': 'prototypes_per_class',
            'x': np.arange(1, 4),
            'y': np.arange(2, 4)
        },
        fixed_params={
            'initial_prototypes': None,
            'max_iter': max_iter,
            'gtol': gtol,
            'random_state': random_state,
            'display': display
        },
        X=X_scaled,
        y=y,
        metric='accuracy',
        title=f'{name}: beta vs prototypes_per_class'
    )
