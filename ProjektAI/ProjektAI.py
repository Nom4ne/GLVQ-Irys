import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel, GmlvqModel, LgmlvqModel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# === PARAMETRY GLOBALNE ===
prototypes_per_class = 25
initial_prototypes   = None
max_iter              = 500
gtol                  = 1e-5
beta                  = 20
random_state          = 42
display               = False
n_thresholds          = 200

# === DANE ===
iris         = load_iris()
X            = iris.data
y            = iris.target
target_names = iris.target_names

# Skalowanie i podział
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=random_state
)

# === FUNKCJE ===
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
    return fpr, tpr, thr

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
    y_bin = label_binarize(y_test, classes=np.arange(n_cls))
    fpr={}; tpr={}; aucv={}
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_on_thresholds(y_bin[:,i], probs[:,i], n_thresh)
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

def train_and_evaluate(model, name):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    acc = accuracy_score(y_test, y_pred)
    prec= precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1  = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {acc:.3f}, Precision(m):{prec:.3f}, Recall(m):{rec:.3f}, F1(m):{f1:.3f}")
    plot_prototypes(model, X_scaled, y, f"{name} – prototypy")
    plot_transformation_matrix(model, f"{name} – transformacja")
    plot_roc_auc_dense(model, X_test, y_test, name, n_thresholds)

# === WYKRES 3D METAPARAMETRÓW ===
def plot_param_dependency_3d(model_class, param1_name, param1_values,
                             param2_name, param2_values, model_name="Model"):
    results = np.zeros((len(param1_values), len(param2_values)))

    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            kwargs = {
                param1_name: val1,
                param2_name: val2,
                "prototypes_per_class": prototypes_per_class,
                "initial_prototypes": initial_prototypes,
                "max_iter": max_iter,
                "random_state": random_state,
                "display": display
            }
            model = model_class(**kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[i, j] = acc
            print(f"{param1_name}={val1}, {param2_name}={val2} => Acc: {acc:.3f}")

    # Tworzenie siatki do wykresu 3D
    P1, P2 = np.meshgrid(param2_values, param1_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P1, P2, results, cmap=cm.viridis, edgecolor='k', alpha=0.8)
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_zlabel("Accuracy")
    ax.set_title(f"{model_name}: Accuracy vs {param1_name} & {param2_name}")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

# === URUCHOMIENIE MODELI ===
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

# === PRZYKŁAD UŻYCIA FUNKCJI 3D ===
# Zależność accuracy od beta i gtol dla GMLVQ

plot_param_dependency_3d(GlvqModel,
    param1_name="beta", param1_values=list(range(2, 20)),
    param2_name="gtol", param2_values=np.logspace(-6, -2, 5),
    model_name="GLVQ"
)

plot_param_dependency_3d(GmlvqModel,
    param1_name="beta", param1_values=list(range(2, 20)),
    param2_name="gtol", param2_values=np.logspace(-6, -2, 5),
    model_name="GMLVQ"
)

plot_param_dependency_3d(LgmlvqModel,
    param1_name="beta", param1_values=list(range(2, 20)),
    param2_name="gtol", param2_values=np.logspace(-6, -2, 5),
    model_name="LGMLVQ"
)