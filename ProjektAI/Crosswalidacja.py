import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
max_iter             = 1000
gtol                 = 1e-5
beta                 = 2
random_state         = 42
display              = False
n_thresholds         = 200
cv_folds             = 25
# === PARAMETRY GLOBALNE Reset ===

def reset_parameters():
    global prototypes_per_class, initial_prototypes, max_iter, gtol, beta, random_state, display, n_thresholds, cv_folds
    prototypes_per_class = 50
    initial_prototypes   = None
    max_iter             = 1000
    gtol                 = 1e-5
    beta                 = 2
    random_state         = 42
    display              = False
    cv_folds             = 25

# === DANE ===
iris         = load_iris()
X            = iris.data
y            = iris.target
target_names = iris.target_names

# Skalowanie wszystkich danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Funkcje wizualizacji  ---

# --- Funkcja treningu i ewaluacji z CV ---
def train_and_evaluate_cv(model, name, X, y, cv=cv_folds):
    logging.info(f"Rozpoczynam generowanie dla modelu: {name}")
    logging.info(f"=== {name} (Cross-Validation: {cv} folds) ===")
    y_pred = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
    acc = accuracy_score(y, y_pred)
    logging.info(f"CV Accuracy: {acc:.3f}")
    logging.info("\n" + classification_report(y, y_pred, target_names=target_names))
    model.fit(X, y)
    
    

# ================================================
#  max_iter vs random_state – surface 3D
# ================================================

def plot_maxiter_randomstate_dependency_3d_cv(model_class, max_iters, random_states, fixed_params, X, y, metric, title, cv=cv_folds):
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}")
    logging.info(f"=== {model_class.__name__} (Cross-Validation: {cv} folds) ===")
    results = []
    start_time = time.time()
    for max_iter in max_iters:
        for random_state in random_states:
            params = fixed_params.copy()
            params['max_iter'] = max_iter
            params['random_state'] = random_state
            model = model_class(**params)

            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                mean_score = np.mean(scores)
            except Exception:
                mean_score = np.nan

            results.append((max_iter, random_state, mean_score))

    # Przekształcanie do siatki
    max_iter_vals = sorted(set(r[0] for r in results))
    random_state_vals = sorted(set(r[1] for r in results))
    Xg, Yg = np.meshgrid(max_iter_vals, random_state_vals)
    Z = np.empty_like(Xg, dtype=float)

    for i, rs in enumerate(random_state_vals):
        for j, mi in enumerate(max_iter_vals):
            for r in results:
                if r[0] == mi and r[1] == rs:
                    Z[i, j] = r[2]
                    break

    # Najlepszy i najgorszy punkt
    valid_results = [r for r in results if not np.isnan(r[2])]
    best = max(valid_results, key=lambda x: x[2])
    worst = min(valid_results, key=lambda x: x[2])

    # Logi tylko dla tych punktów
    logging.info(f"[BEST] max_iter={best[0]}, random_state={best[1]}, {metric}={best[2]:.4f}")
    logging.info(f"[WORST] max_iter={worst[0]}, random_state={worst[1]}, {metric}={worst[2]:.4f}")

    # Wykres
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Z, cmap=cm.viridis, edgecolor='k', alpha=0.85, vmin=0.8, vmax=1.0)

    ax.scatter(best[0], best[1], best[2], color='red', s=100, label='Najlepszy')
    ax.scatter(worst[0], worst[1], worst[2], color='black', s=100, label='Najgorszy')

    ax.set_xlabel('max_iter')
    ax.set_ylabel('random_state')
    ax.set_zlabel(metric)
    ax.set_title(title)
    ax.set_zlim(0.0, 1.0)
    ax.legend()

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric)
    save_current_fig(title)

    # Widok z góry
    ax.view_init(elev=90, azim=-90)
    save_current_fig(f"{title}_topview")
    plt.close()

    elapsed = time.time() - start_time
    logging.info(f"Czas wykonania: {elapsed:.2f} sekundy")

    return {
        'results': results,
        'best': {
            'max_iter': best[0],
            'random_state': best[1],
            metric: best[2]
        },
        'worst': {
            'max_iter': worst[0],
            'random_state': worst[1],
            metric: worst[2]
        },
        'time_sec': elapsed
    }


# ================================================
#  beta vs prot_per_class surface 3D
# ================================================
def plot_param_dependency_3d_cv(model_class, param_grid, fixed_params, X, y,
                                metric='accuracy', title="Wykres_3D", cv=cv_folds):
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}")
    logging.info(f"=== {model_class} (Cross-Validation: {cv} folds) ===")
    start_time = time.time()

    xs, ys = list(param_grid['x']), list(param_grid['y'])
    Xp, Yp = np.meshgrid(xs, ys)
    Z = np.zeros_like(Xp, dtype=float)

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            params = fixed_params.copy()
            params[param_grid['x_name']] = int(xv) if param_grid['x_name'] == 'beta' else xv
            params[param_grid['y_name']] = int(yv) if param_grid['y_name'] == 'beta' else yv
            try:
                model = model_class(**params)
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                Z[j, i] = scores.mean()
            except Exception as e:
                logging.warning(f"Błąd dla parametrów {params}: {e}")
                Z[j, i] = np.nan

    elapsed = time.time() - start_time

    # Szukanie najlepszego i najgorszego punktu
    if np.isnan(Z).all():
        logging.error("Wszystkie wyniki to NaN — przerwano")
        return None

    best_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    worst_idx = np.unravel_index(np.nanargmin(Z), Z.shape)

    best_x, best_y = Xp[best_idx], Yp[best_idx]
    best_score = Z[best_idx]
    worst_x, worst_y = Xp[worst_idx], Yp[worst_idx]
    worst_score = Z[worst_idx]

    # Podsumowanie
    logging.info("=== Podsumowanie wyników ===")
    logging.info(f"Najlepszy wynik {metric}: {best_score:.4f} przy {param_grid['x_name']}={best_x}, {param_grid['y_name']}={best_y}")
    logging.info(f"Najgorszy wynik {metric}: {worst_score:.4f} przy {param_grid['x_name']}={worst_x}, {param_grid['y_name']}={worst_y}")
    logging.info(f"Czas zbierania danych: {elapsed:.2f} sekundy")
    logging.info("============================")

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xp, Yp, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel(param_grid['x_name'])
    ax.set_ylabel(param_grid['y_name'])
    ax.set_zlabel(metric)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.scatter(best_x, best_y, best_score, color='r', s=100, label='Najlepszy punkt')
    ax.scatter(worst_x, worst_y, worst_score, color='black', s=100, label='Najgorszy punkt')
    ax.legend()
    save_current_fig(title)

    # Widok z góry
    ax.view_init(elev=90, azim=-90)
    save_current_fig(f"{title}_topview")
    plt.close()

    return {
        param_grid['x_name']: best_x,
        param_grid['y_name']: best_y,
        metric: best_score,
        'worst_' + param_grid['x_name']: worst_x,
        'worst_' + param_grid['y_name']: worst_y,
        'worst_' + metric: worst_score,
        'time_sec': round(elapsed, 2)
    }


# ================================================
# --- beta vs gtol 3D  – surface 3D
# ================================================
def plot_beta_gtol_dependency_3d_cv(model_class, betas, gtol_values, fixed_params,
                                   X, y, metric='accuracy', title="beta_vs_gtol_cv", cv=cv_folds):
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}")
    logging.info(f"=== {model_class} (Cross-Validation: {cv} folds) ===")
    
    log_gtol_values = np.log10(gtol_values)
    B, G_log = np.meshgrid(betas, log_gtol_values)
    Z = np.zeros_like(B, dtype=float)

    start_time = time.time()
    for i, b in enumerate(betas):
        for j, g in enumerate(gtol_values):
            params = fixed_params.copy()
            params['beta'] = int(b)
            params['gtol'] = float(g)
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            Z[j, i] = scores.mean()
    
    elapsed = time.time() - start_time

    # Najlepszy punkt
    idx_best = np.unravel_index(np.argmax(Z), Z.shape)
    best_beta, best_gtol = B[idx_best], gtol_values[idx_best[0]]
    best_score = Z[idx_best]
    logging.info(f"Najlepszy {metric}={best_score:.4f} przy beta={best_beta}, gtol={best_gtol}")

    # Najgorszy punkt
    idx_worst = np.unravel_index(np.argmin(Z), Z.shape)
    worst_beta, worst_gtol = B[idx_worst], gtol_values[idx_worst[0]]
    worst_score = Z[idx_worst]
    logging.info(f"Najgorszy {metric}={worst_score:.4f} przy beta={worst_beta}, gtol={worst_gtol}")

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B, G_log, Z, cmap='viridis', edgecolor='k', alpha=0.8)

    # Dodanie najlepszego punktu (czerwona kula)
    ax.scatter(best_beta, np.log10(best_gtol), best_score, color='red', s=100, label='Najlepszy punkt')

    # Dodanie najgorszego punktu (czarna kula)
    ax.scatter(worst_beta, np.log10(worst_gtol), worst_score, color='black', s=100, label='Najgorszy punkt')

    ax.set_xlabel('beta')
    ax.set_ylabel('log10(gtol)')
    ax.set_zlabel(metric)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.legend()
    save_current_fig(title)

    # Widok z góry
    ax.view_init(elev=90, azim=-90)
    save_current_fig(f"{title}_topview")
    plt.close()

    return {
        'beta': best_beta, 
        'gtol': best_gtol, 
        metric: best_score, 
        'worst_beta': worst_beta,
        'worst_gtol': worst_gtol,
        f'worst_{metric}': worst_score,
        'time_sec': elapsed
    }


# === URUCHOMIENIE ===

# Base params
base_params = {
               'prototypes_per_class':prototypes_per_class,
               'initial_prototypes':initial_prototypes,
               'max_iter':max_iter,
               'gtol':gtol,
               'beta':int(beta),
               'random_state':random_state,
               'display':display
               }

models = {'GLVQ':GlvqModel}

# === Wywołanie dla Iris ===
# ================================================
# === SEKCJA URUCHOMIENIA ===
# ================================================
# Zakresy parametrów
max_iters     = [5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225]
random_states = [0, 10, 25, 50, 75,100]



# Wywołanie funkcji 
for name, cls in models.items():
    res1 = plot_maxiter_randomstate_dependency_3d_cv(
        model_class=cls,
        max_iters=max_iters,
        random_states=random_states,
        fixed_params=base_params,
        X=X_scaled,
        y=y,
        metric='accuracy',
        title=f"{name}_maxIter_vs_randomState",
        cv=cv_folds
    )

reset_parameters()

# CV trening i ewaluacja
for name, cls in models.items():
    train_and_evaluate_cv(cls(**base_params), name, X_scaled, y, cv=cv_folds)

reset_parameters()

# CV siatka beta vs prototypes_per_class
for name, cls in models.items():
    res = plot_param_dependency_3d_cv(
        cls, {'x_name':'beta','y_name':'prototypes_per_class','x':range(2,26),'y':range(2,26)},
        base_params, X_scaled, y, 'accuracy', f"{name}_beta_vs_prototypes_cv", cv=cv_folds
    )
    logging.info(f"Wyniki {name} beta_vs_prototypes CV: {res}")

reset_parameters()

# CV beta vs gtol
betas = list(range(2,21))
gtol_vals = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for name, cls in models.items():
    res = plot_beta_gtol_dependency_3d_cv(
        cls, betas, gtol_vals, base_params, X_scaled, y, 'accuracy', f"{name}_beta_vs_gtol_cv", cv=cv_folds
    )
    logging.info(f"Wyniki {name} beta_vs_gtol CV: {res}")