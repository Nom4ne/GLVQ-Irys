import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel, GmlvqModel, LgmlvqModel
from mpl_toolkits.mplot3d import Axes3D

# === KONFIGURACJA LOGOWANIA ===
log_filename = 'console.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)

# folder do zapisu grafik
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

def save_current_fig(title):
    safe_title = title.replace(' ', '_').replace('–', '-')
    filename = os.path.join(figures_dir, f"{safe_title}.png")
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    logging.info(f"Zapisano grafikę: {filename}")

# === PARAMETRY GLOBALNE ===
prototypes_per_class = 50
initial_prototypes   = None
max_iter             = 5000
gtol                 = 1e-5
beta                 = 2
random_state         = 42
display              = False
n_thresholds         = 200

# === DANE ===
iris         = load_iris()
X            = iris.data
y            = iris.target
target_names = iris.target_names

# Skalowanie i podział
scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=random_state
)

# --- Funkcje pomocnicze i wizualizacje ---

def plot_prototypes(model, X, y, title):
    pca = PCA(2)
    Xp = pca.fit_transform(X)
    try:
        P = pca.transform(model.w_)
        labels = model.c_w_
    except AttributeError:
        logging.warning(f"{type(model).__name__}: brak prototypów.")
        return
    plt.figure(figsize=(7,5))
    plt.scatter(Xp[:,0], Xp[:,1], c=y, cmap='viridis', alpha=0.4)
    plt.scatter(P[:,0], P[:,1], c=labels, cmap='tab10', marker='X', s=150, edgecolor='k')
    plt.title(title)
    plt.grid(True)
    save_current_fig(title)
    if display: plt.show()
    plt.close()


def plot_transformation_matrix(model, title):
    logging.info(f"Rysowanie macierzy transformacji dla: {title}")
    if hasattr(model, 'omega_'):
        M = model.omega_
        mat = M.T @ M
    elif hasattr(model, 'omegas_'):
        O = model.omegas_[0]
        mat = O.T @ O
        title += " (1. prototyp)"
    else:
        logging.warning(f"{type(model).__name__}: brak transformacji.")
        return
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Cecha")
    plt.ylabel("Cecha")
    save_current_fig(title)
    if display: plt.show()
    plt.close()


def roc_on_thresholds(y_true_bin, scores, n_thresh=100):
    thr = np.linspace(0, 1, n_thresh)
    tpr = np.zeros(n_thresh)
    fpr = np.zeros(n_thresh)
    P = y_true_bin.sum()
    N = len(y_true_bin) - P
    for i, t in enumerate(thr):
        y_pred = (scores >= t).astype(int)
        tp = np.logical_and(y_pred == 1, y_true_bin == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true_bin == 0).sum()
        tpr[i] = tp / P if P > 0 else 0
        fpr[i] = fp / N if N > 0 else 0
    return fpr, tpr


def plot_roc_auc_dense(model, X_test, y_test, name, n_thresh=100):
    logging.info(f"Rysowanie ROC dla: {name}")
    W = model.w_
    if hasattr(model, 'omega_'):
        Xp = X_test @ model.omega_.T
        Wp = W @ model.omega_.T
    elif hasattr(model, 'omegas_'):
        O = model.omegas_[0]
        Xp = X_test @ O.T
        Wp = W @ O.T
    else:
        Xp, Wp = X_test, W
    d = np.linalg.norm(Xp[:, None, :] - Wp[None, :, :], axis=2)
    inv = 1 / (d + 1e-12)
    proto_p = inv / inv.sum(1, keepdims=True)
    n_cls = len(target_names)
    probs = np.zeros((len(y_test), n_cls))
    for cls in range(n_cls):
        probs[:, cls] = proto_p[:, model.c_w_ == cls].sum(1)
    fpr, tpr, aucv = {}, {}, {}
    for i in range(n_cls):
        fpr[i], tpr[i] = roc_on_thresholds((y_test == i).astype(int), probs[:, i], n_thresh)
        idx = np.argsort(fpr[i])
        aucv[i] = np.trapz(tpr[i][idx], fpr[i][idx])
    macro_auc = np.mean(list(aucv.values()))
    logging.info(f"AUC-ROC {name}: {aucv}, Macro AUC: {macro_auc:.3f}")
    plt.figure(figsize=(7,5))
    for i, col in zip(range(n_cls), ['r','g','b']):
        plt.plot(fpr[i], tpr[i], col, lw=2, label=f"{target_names[i]} (AUC={aucv[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC – {name}")
    plt.legend(loc='lower right')
    plt.grid(True)
    save_current_fig(f"ROC_{name}")
    if display: plt.show()
    plt.close()


def plot_classification_metrics(report_dict, name):
    logging.info(f"Rysowanie metryk klasyfikacji dla: {name}")
    classes = [c for c in report_dict if c in target_names]
    metrics = ['precision', 'recall', 'f1-score']
    values = {m: [report_dict[c][m] for c in classes] for m in metrics}
    x = np.arange(len(classes))
    width = 0.2
    plt.figure(figsize=(8,5))
    for i, m in enumerate(metrics):
        plt.bar(x + i*width, values[m], width=width, label=m)
    plt.xticks(x + width, classes)
    plt.ylim(0,1)
    plt.ylabel('Score')
    plt.title(f'Metrics_{name}')
    plt.legend()
    save_current_fig(f"Metrics_{name}")
    if display: plt.show()
    plt.close()


def train_and_evaluate(model, name):
    logging.info(f"=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    logging.info("\n" + classification_report(y_test, y_pred, target_names=target_names))
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.3f}")
    plot_prototypes(model, X_scaled, y, f"{name} – prototypy")
    plot_transformation_matrix(model, f"{name} – transformacja")
    plot_roc_auc_dense(model, X_test, y_test, name, n_thresholds)
    plot_classification_metrics(report, name)

# === Stara funkcja 3D (parametry x/y) ===
def plot_param_dependency_3d(model_class, param_grid, fixed_params, X, y,
                             metric='accuracy', title="Wykres_3D"):
    start_time = time.time()
    xs = list(param_grid['x'])
    ys = list(param_grid['y'])
    Xp, Yp = np.meshgrid(xs, ys)
    Z = np.zeros_like(Xp, dtype=float)
    for i, x_val in enumerate(xs):
        for j, y_val in enumerate(ys):
            params = fixed_params.copy()
            params['prototypes_per_class'] = int(params.get('prototypes_per_class', prototypes_per_class))
            params['initial_prototypes'] = initial_prototypes
            if param_grid['x_name']=='beta': params['beta']=int(x_val)
            else: params[param_grid['x_name']]=x_val
            if param_grid['y_name']=='beta': params['beta']=int(y_val)
            else: params[param_grid['y_name']]=y_val
            model = model_class(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            Z[j,i] = accuracy_score(y, y_pred)
    elapsed = time.time() - start_time
    logging.info(f"Czas trenowania i generowania danych: {elapsed:.2f} sekundy")
    idx_flat = np.argmax(Z)
    j_best, i_best = np.unravel_index(idx_flat, Z.shape)
    best_x, best_y = Xp[j_best,i_best], Yp[j_best,i_best]
    best_score = Z[j_best,i_best]
    logging.info(f"Najlepszy wynik {metric}: {best_score:.4f} przy {param_grid['x_name']}={best_x}, {param_grid['y_name']}={best_y}")
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xp, Yp, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel(param_grid['x_name'])
    ax.set_ylabel(param_grid['y_name'])
    ax.set_zlabel(metric)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric)
    ax.scatter(best_x, best_y, best_score, color='r', s=100, label='Najlepszy punkt')
    ax.legend()
    # standard
    save_current_fig(title)
    # topview
    ax.view_init(elev=90, azim=-90)
    save_current_fig(f"{title}_topview")
    if display: plt.show()
    plt.close()
    return {param_grid['x_name']:best_x, param_grid['y_name']:best_y, metric:best_score, 'time_sec':elapsed}

# === Funkcja 3D beta vs gtol ===
def plot_beta_gtol_dependency_3d(model_class, betas, gtol_values, fixed_params,
                                 X, y, metric='accuracy', title="beta_vs_gtol"):
    start_time = time.time()
    B, G = np.meshgrid(betas, gtol_values)
    Z = np.zeros_like(B, dtype=float)
    for i, beta_val in enumerate(betas):
        for j, gtol_val in enumerate(gtol_values):
            params = fixed_params.copy()
            params['beta'] = int(beta_val)
            params['gtol'] = float(gtol_val)
            model = model_class(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            Z[j, i] = accuracy_score(y, y_pred)
    elapsed = time.time() - start_time
    idx = np.unravel_index(np.argmax(Z), Z.shape)
    best_beta = B[idx]
    best_gtol = G[idx]
    best_score = Z[idx]
    logging.info(f"Najlepszy accuracy={best_score:.4f} przy beta={best_beta}, gtol={best_gtol}")
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B, G, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('beta')
    ax.set_ylabel('gtol')
    ax.set_zlabel(metric)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    save_current_fig(title)
    ax.view_init(elev=90, azim=-90)
    save_current_fig(f"{title}_topview")
    if display: plt.show()
    plt.close()
    return {'beta': best_beta, 'gtol': best_gtol, metric: best_score, 'time_sec': elapsed}

# === Badanie wpływu gtol ===
def evaluate_gtol_dependency_for_all(models, gtol_values, fixed_params, X, y,
                                     metric='accuracy', plot=True):
    results_all = {}
    for name, model_class in models.items():
        logging.info(f"==> Badanie wpływu gtol dla modelu: {name}")
        start_time = time.time()
        results = []
        for g in gtol_values:
            params = fixed_params.copy()
            params['gtol'] = float(g)
            model = model_class(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            score = accuracy_score(y, y_pred)
            results.append((g, score))
        elapsed = time.time() - start_time
        results = np.array(results)
        best_idx = np.argmax(results[:, 1])
        best_g, best_score = results[best_idx]
        logging.info(f"  Najlepszy {metric}: {best_score:.4f} przy gtol={best_g}")
        logging.info(f"  Czas: {elapsed:.2f} s")
        if plot:
            plt.figure(figsize=(8,5))
            plt.plot(results[:, 0], results[:, 1], marker='o', label=f'{name}')
            plt.axvline(best_g, color='r', linestyle='--', label=f'Najlepszy gtol: {best_g}')
            plt.xscale('log')
            plt.title(f"{name}: Wpływ gtol na {metric}")
            plt.xlabel('gtol')
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            save_current_fig(f"gtol_{name}")
            if display: plt.show()
            plt.close()
        results_all[name] = {'gtol': best_g, metric: best_score, 'time_sec': elapsed}
    return results_all

# === INICJALIZACJA I URUCHOMIENIE ===
base_params = {'prototypes_per_class':prototypes_per_class,
               'initial_prototypes':initial_prototypes,
               'max_iter':max_iter,'gtol':gtol,
               'beta':int(beta),'random_state':random_state,'display':display}
models = {'GLVQ':GlvqModel,'GMLVQ':GmlvqModel,'LGMLVQ':LgmlvqModel}

# Trening i ewaluacja
for name, cls in models.items():
    model = cls(**base_params)
    train_and_evaluate(model, name)
    # Siatka x/y (starsza funkcja)
    best_xy = plot_param_dependency_3d(
        cls, {'x_name':'beta','y_name':'prototypes_per_class','x':range(2,51),'y':range(2,51)},
        base_params, X_scaled, y, 'accuracy', f"{name}_beta_vs_prototypes"
    )
    logging.info(f"Wyniki param_grid dla {name}: {best_xy}")



# Wartości parametrów
betas = list(range(2, 21))
gtol_vals = [1e-10, 1e-9 ,1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

# GLVQ
best_bg_glvq = plot_beta_gtol_dependency_3d(
    GlvqModel, betas, gtol_vals, base_params, X_scaled, y,
    metric='accuracy', title="GLVQ_beta_vs_gtol"
)
logging.info(f"Wynik beta/gtol dla GLVQ: {best_bg_glvq}")

# GMLVQ
best_bg_gmlvq = plot_beta_gtol_dependency_3d(
    GmlvqModel, betas, gtol_vals, base_params, X_scaled, y,
    metric='accuracy', title="GMLVQ_beta_vs_gtol"
)
logging.info(f"Wynik beta/gtol dla GMLVQ: {best_bg_gmlvq}")

# LGMLVQ
best_bg_lgmlvq = plot_beta_gtol_dependency_3d(
    LgmlvqModel, betas, gtol_vals, base_params, X_scaled, y,
    metric='accuracy', title="LGMLVQ_beta_vs_gtol"
)
logging.info(f"Wynik beta/gtol dla LGMLVQ: {best_bg_lgmlvq}")


# Badanie wpływu gtol dla wszystkich modeli
gtol_values = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
gtol_results = evaluate_gtol_dependency_for_all(
    models=models,
    gtol_values=gtol_values,
    fixed_params=base_params,
    X=X_scaled,
    y=y,
    metric='accuracy',
    plot=True
)
logging.info("== Podsumowanie wyników dla wszystkich modeli ==")
for model_name, result in gtol_results.items():
    logging.info(f"{model_name}: gtol={result['gtol']}, accuracy={result['accuracy']:.4f}, czas={result['time_sec']:.2f}s")

# Test jednostkowy
def test_beta_gtol_dependency():
    # Parametry bazowe do modelu
    test_params = {
        'prototypes_per_class': 5,
        'initial_prototypes': None,
        'max_iter': 500,
        'random_state': 0,
        'display': False
    }

    # Zakresy parametrów do testu
    betas_test = list(range(2, 12))  # beta: 2–11
    gtol_test = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]  # różne wartości gtol

    # Wywołanie funkcji do wykresu 3D i uzyskanie najlepszego wyniku
    res = plot_param_dependency_3d(
        GlvqModel,
        {'x_name': 'beta', 'y_name': 'gtol', 'x': betas_test, 'y': gtol_test},
        test_params,
        X_scaled,
        y,
        metric='accuracy',
        title="test_beta_vs_gtol"
    )

    # Sprawdzenia poprawności wyników
    assert 'beta' in res and 'gtol' in res and 'accuracy' in res, "Brakuje kluczy w wyniku"
    assert res['beta'] in betas_test, "Najlepszy beta spoza testowanego zakresu"
    assert res['gtol'] in gtol_test, "Najlepszy gtol spoza testowanego zakresu"
    assert 0.0 <= res['accuracy'] <= 1.0, "Nieprawidłowa dokładność"

    print("test_beta_gtol_dependency passed.")


test_beta_gtol_dependency()
