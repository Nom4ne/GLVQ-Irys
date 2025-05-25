import time # Importuje moduł 'time', który dostarcza funkcji do pracy z czasem, np. mierzenia czasu wykonania.
import numpy as np # Importuje bibliotekę NumPy, powszechnie używaną do numerycznych obliczeń i operacji na tablicach wielowymiarowych, nadając jej alias 'np'.
import matplotlib.pyplot as plt # Importuje moduł 'pyplot' z biblioteki Matplotlib do tworzenia wykresów i wizualizacji, nadając mu alias 'plt'.
import logging # Importuje moduł 'logging' do konfigurowania i generowania komunikatów logów.
import os # Importuje moduł 'os', który umożliwia interakcję z systemem operacyjnym, np. tworzenie katalogów.
from sklearn.datasets import load_iris # Importuje funkcję 'load_iris' z modułu 'datasets' scikit-learn do wczytywania standardowego zbioru danych Iris.
from sklearn.model_selection import cross_val_score, cross_val_predict # Importuje funkcje do walidacji krzyżowej: 'cross_val_score' (ocena) i 'cross_val_predict' (generowanie przewidywań).
from sklearn.preprocessing import StandardScaler # Importuje 'StandardScaler' do standaryzacji danych (skalowania cech do średniej 0 i wariancji 1).
from sklearn.metrics import classification_report, accuracy_score # Importuje metryki do oceny modeli klasyfikacji: 'classification_report' (szczegółowy raport) i 'accuracy_score' (dokładność).
from sklearn_lvq import GlvqModel # Importuje klasę 'GlvqModel' z biblioteki sklearn-lvq, która implementuje model Generalized Learning Vector Quantization.
from mpl_toolkits.mplot3d import Axes3D # Importuje 'Axes3D' z 'mplot3d' (rozszerzenie Matplotlib) do tworzenia trójwymiarowych wykresów.
from matplotlib import cm # Importuje moduł 'cm' (colormaps) z Matplotlib, zawierający kolekcje map kolorów do wizualizacji danych.

# === KONFIGURACJA LOGOWANIA ===
log_filename = 'console.log' # Definiuje nazwę pliku, do którego będą zapisywane logi.
logging.basicConfig( # Konfiguruje podstawowe ustawienia systemu logowania.
    level=logging.INFO, # Ustawia poziom logowania na INFO, co oznacza, że komunikaty o poziomie INFO i wyższym będą rejestrowane.
    format='%(asctime)s - %(levelname)s - %(message)s', # Definiuje format komunikatów logowania (czas, poziom, wiadomość).
    handlers=[ # Określa, gdzie komunikaty logowania mają być wysyłane.
        logging.FileHandler(log_filename, mode='w'), # Handler zapisujący logi do pliku 'console.log', nadpisując go przy każdym uruchomieniu ('w').
        logging.StreamHandler() # Handler wyświetlający logi w konsoli.
    ]
)

# folder do zapisu grafik
figures_dir = 'figures' # Definiuje nazwę katalogu, w którym będą zapisywane wygenerowane wykresy.
os.makedirs(figures_dir, exist_ok=True) # Tworzy katalog 'figures', jeśli jeszcze nie istnieje; 'exist_ok=True' zapobiega błędowi, jeśli katalog już jest.

def save_current_fig(title): # Definiuje funkcję do zapisywania bieżącego wykresu Matplotlib do pliku PNG.
    safe_title = title.replace(' ', '_').replace('–', '-') # Zastępuje spacje i myślniki w tytule podkreśleniami, tworząc bezpieczną nazwę pliku.
    filename = os.path.join(figures_dir, f"{safe_title}.png") # Tworzy pełną ścieżkę do pliku graficznego w katalogu 'figures'.
    fig = plt.gcf() # Pobiera referencję do aktualnie aktywnej figury (wykresu).
    fig.set_size_inches(20, 12) # Ustawia rozmiar figury w calach na 20x12, zapewniając większy obraz.
    plt.savefig(filename, bbox_inches='tight', dpi=300) # Zapisuje wykres do pliku: 'bbox_inches='tight'' przycina białe marginesy, 'dpi=300' ustawia wysoką rozdzielczość.
    logging.info(f"Zapisano grafikę: {filename}") # Loguje informację o zapisaniu pliku graficznego.

# === PARAMETRY GLOBALNE ===
# Consolidated base parameters for models
BASE_PARAMS = { # Definiuje słownik z podstawowymi parametrami do inicjalizacji modeli LVQ.
    'prototypes_per_class': 5, # Liczba prototypów do wygenerowania dla każdej klasy.
    'initial_prototypes': None, # Opcja początkowych prototypów (None oznacza domyślne inicjalizowanie).
    'max_iter': 1000, # Maksymalna liczba iteracji algorytmu optymalizacji.
    'gtol': 1e-5, # Tolerancja gradientu dla kryterium zbieżności.
    'beta': 2, # Parametr regularyzacji beta (ważny w LVQ).
    'random_state': 42, # Ziarno generatora liczb pseudolosowych dla powtarzalności wyników.
    'display': False # Czy wyświetlać postęp optymalizacji (ustawione na False dla cichego działania).
}
CV_FOLDS = 5 # Liczba fałd (k-krotności) do walidacji krzyżowej.

def reset_parameters(): # Definiuje funkcję, która resetuje globalne parametry do ich początkowych wartości.
    """Resets global parameters to their initial values. # Docstring: Krótki opis funkcji.
    This function is primarily for demonstration or specific testing scenarios
    where global parameter values need to be reverted.
    """
    global BASE_PARAMS, CV_FOLDS # Deklaruje, że zmienne BASE_PARAMS i CV_FOLDS są globalne i będą modyfikowane.
    BASE_PARAMS = { # Resetuje słownik BASE_PARAMS do jego oryginalnych wartości.
        'prototypes_per_class': 5,
        'initial_prototypes': None,
        'max_iter': 1000,
        'gtol': 1e-5,
        'beta': 2,
        'random_state': 42,
        'display': False
    }
    CV_FOLDS = 5 # Resetuje liczbę fałd walidacji krzyżowej.


# === DANE ===
iris = load_iris() # Wczytuje zbiór danych Iris.
X = iris.data # Przypisuje dane cech (input features) ze zbioru Iris do zmiennej X.
y = iris.target # Przypisuje etykiety klas (target labels) ze zbioru Iris do zmiennej y.
target_names = iris.target_names # Przypisuje nazwy klas (np. 'setosa', 'versicolor') do zmiennej target_names.

# Skalowanie wszystkich danych
scaler = StandardScaler() # Tworzy instancję obiektu StandardScaler, który będzie używany do standaryzacji danych.
X_scaled = scaler.fit_transform(X) # Standaryzuje dane X: najpierw dopasowuje scaler do danych, a następnie je przekształca, by miały średnią 0 i wariancję 1.

# --- Funkcja treningu i ewaluacji z CV ---
def train_and_evaluate_cv(model_class, name, X, y, cv=CV_FOLDS, params=None): # Definiuje funkcję do trenowania i oceny modelu z walidacją krzyżową.
    if params is None: # Sprawdza, czy podano konkretne parametry dla modelu.
        params = BASE_PARAMS.copy() # Jeśli nie, używa kopii globalnych podstawowych parametrów.
    model = model_class(**params) # Tworzy instancję modelu (np. GlvqModel) z podanymi parametrami.

    logging.info(f"Rozpoczynam generowanie dla modelu: {name}") # Loguje informację o rozpoczęciu procesu dla danego modelu.
    logging.info(f"=== {name} (Cross-Validation: {cv} folds) ===") # Loguje nagłówek z nazwą modelu i liczbą fałd walidacji krzyżowej.
    y_pred = cross_val_predict(model, X, y, cv=cv) # Wykonuje walidację krzyżową i zwraca przewidywania dla każdego elementu zbioru danych.
    acc = accuracy_score(y, y_pred) # Oblicza dokładność (accuracy) przewidywań w porównaniu z rzeczywistymi etykietami.
    logging.info(f"CV Accuracy: {acc:.3f}") # Loguje obliczoną dokładność walidacji krzyżowej z trzema miejscami po przecinku.
    logging.info("\n" + classification_report(y, y_pred, target_names=target_names)) # Generuje i loguje szczegółowy raport klasyfikacji (precyzja, kompletność, F1-score).
    # It's good practice to fit the model on the full dataset after CV for final use
    model.fit(X, y) # Po walidacji krzyżowej, model jest dopasowywany do całego zbioru danych w celu jego ostatecznego użycia.

# ================================================
# max_iter vs random_state – surface 3D
# ================================================

def plot_maxiter_randomstate_dependency_3d_cv(model_class, max_iters, random_states, fixed_params, X, y, metric, title, cv=CV_FOLDS): # Definiuje funkcję do tworzenia wykresu 3D zależności metryki od max_iter i random_state.
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}") # Loguje nazwę modelu, dla którego generowany jest wykres.
    logging.info(f"=== {model_class.__name__} (Cross-Validation: {cv} folds) ===") # Loguje nagłówek dla walidacji krzyżowej.
    results = [] # Inicjuje pustą listę do przechowywania wyników dla różnych kombinacji parametrów.
    start_time = time.time() # Zapisuje czas rozpoczęcia zbierania danych.
    for max_iter_val in max_iters: # Iteruje przez każdą wartość z listy max_iters.
        for random_state_val in random_states: # Iteruje przez każdą wartość z listy random_states.
            params = fixed_params.copy() # Kopiuje stałe parametry, aby nie modyfikować oryginalnego słownika.
            params['max_iter'] = max_iter_val # Ustawia aktualną wartość max_iter.
            params['random_state'] = random_state_val # Ustawia aktualną wartość random_state.
            model = model_class(**params) # Tworzy instancję modelu z bieżącymi parametrami.
            try: # Rozpoczyna blok do obsługi potencjalnych błędów.
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric) # Wykonuje walidację krzyżową i zbiera wyniki dla określonej metryki.
                mean_score = np.mean(scores) # Oblicza średni wynik z walidacji krzyżowej.
            except Exception as e: # Przechwytuje wszelkie wyjątki, które mogą wystąpić.
                logging.warning(f"Błąd dla max_iter={max_iter_val}, random_state={random_state_val}: {e}") # Loguje ostrzeżenie w przypadku błędu.
                mean_score = np.nan # Ustawia wynik na NaN (Not a Number), jeśli wystąpił błąd.

            results.append((max_iter_val, random_state_val, mean_score)) # Dodaje kombinację parametrów i średni wynik do listy wyników.

    # Przekształcanie do siatki
    max_iter_vals_unique = sorted(set(r[0] for r in results)) # Tworzy posortowaną listę unikalnych wartości max_iter.
    random_state_vals_unique = sorted(set(r[1] for r in results)) # Tworzy posortowaną listę unikalnych wartości random_state.\

    Xg, Yg = np.meshgrid(max_iter_vals_unique, random_state_vals_unique) # Tworzy siatkę punktów (Xg, Yg) z unikalnych wartości max_iter i random_state.
    Z = np.empty_like(Xg, dtype=float) # Inicjuje pustą tablicę Z o takim samym kształcie jak Xg, do przechowywania wyników.
    # Populate Z with scores
    print(Xg,Yg,Z)
    print(results)
    for i, rs in enumerate(random_state_vals_unique): # Iteruje przez unikalne wartości random_state.
        for j, mi in enumerate(max_iter_vals_unique): # Iteruje przez unikalne wartości max_iter.
            for r in results: # Przegląda wszystkie zebrane wyniki.
                if r[0] == mi and r[1] == rs: # Jeśli bieżąca kombinacja max_iter i random_state pasuje do wyniku.
                    Z[i, j] = r[2] # Przypisuje wynik (metrykę) do odpowiedniej pozycji w macierzy Z.
                    break # Przerywa wewnętrzną pętlę, ponieważ wynik został znaleziony.
    print(Z)
    # Najlepszy i najgorszy punkt
    valid_results = [r for r in results if not np.isnan(r[2])] # Filtruje wyniki, usuwając te, które są NaN (nieudane).
    if not valid_results: # Sprawdza, czy są jakiekolwiek prawidłowe wyniki do wykreślenia.
        logging.error("No valid results for plotting. Skipping 3D plot.") # Loguje błąd, jeśli nie ma prawidłowych wyników.
        return None # Zwraca None, jeśli nie ma danych do wykresu.

    best = max(valid_results, key=lambda x: x[2]) # Znajduje wynik z najwyższą wartością metryki (najlepszy wynik).
    worst = min(valid_results, key=lambda x: x[2]) # Znajduje wynik z najniższą wartością metryki (najgorszy wynik).

    # Logi tylko dla tych punktów
    logging.info(f"[BEST] max_iter={best[0]}, random_state={best[1]}, {metric}={best[2]:.4f}") # Loguje szczegóły najlepszego wyniku.
    logging.info(f"[WORST] max_iter={worst[0]}, random_state={worst[1]}, {metric}={worst[2]:.4f}") # Loguje szczegóły najgorszego wyniku.

    # Wykres
    fig = plt.figure(figsize=(12, 10)) # Tworzy nową figurę Matplotlib o określonym rozmiarze.
    ax = fig.add_subplot(111, projection='3d') # Dodaje trójwymiarową oś (subplot) do figury.
    surf = ax.plot_surface(Xg, Yg, Z, cmap=cm.viridis, edgecolor='k', alpha=0.85, vmin=0.8, vmax=1.0) # Tworzy powierzchnię 3D: Xg, Yg, Z to współrzędne, 'cmap' to mapa kolorów, 'edgecolor' to kolor krawędzi, 'alpha' to przezroczystość, 'vmin/vmax' to zakres mapowania kolorów.

    ax.scatter(best[0], best[1], best[2], color='red', s=100, label='Najlepszy') # Dodaje czerwony punkt reprezentujący najlepszy wynik.
    ax.scatter(worst[0], worst[1], worst[2], color='black', s=100, label='Najgorszy') # Dodaje czarny punkt reprezentujący najgorszy wynik.

    ax.set_xlabel('max_iter') # Ustawia etykietę osi X.
    ax.set_ylabel('random_state') # Ustawia etykietę osi Y.
    ax.set_zlabel(metric) # Ustawia etykietę osi Z na nazwę metryki.
    ax.set_title(title) # Ustawia tytuł wykresu.
    ax.set_zlim(0.0, 1.0) # Ustawia zakres osi Z od 0.0 do 1.0 (typowy dla dokładności).
    ax.legend() # Wyświetla legendę na wykresie.

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=metric) # Dodaje pasek koloru (colorbar) do wykresu, pokazujący skalę metryki.
    save_current_fig(title) # Zapisuje bieżący wykres do pliku.

    # Widok z góry
    ax.view_init(elev=90, azim=-90) # Zmienia kąt widzenia wykresu na widok z góry (elewacja 90 stopni, azymut -90 stopni).
    save_current_fig(f"{title}_topview") # Zapisuje wykres z widokiem z góry.
    plt.close() # Zamyka bieżącą figurę wykresu, aby zwolnić pamięć.

    elapsed = time.time() - start_time # Oblicza czas, jaki upłynął od rozpoczęcia funkcji.
    logging.info(f"Czas wykonania: {elapsed:.2f} sekundy") # Loguje czas wykonania funkcji.

    return { # Zwraca słownik zawierający wyniki, w tym najlepsze i najgorsze kombinacje parametrów oraz czas wykonania.
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
# beta vs prot_per_class surface 3D
# ================================================
def plot_param_dependency_3d_cv(model_class, param_grid, fixed_params, X, y, # Definiuje bardziej ogólną funkcję do tworzenia wykresu 3D zależności metryki od dwóch dowolnych parametrów.
                                 metric='accuracy', title="Wykres_3D", cv=CV_FOLDS):
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}") # Loguje nazwę modelu.
    logging.info(f"=== {model_class.__name__} (Cross-Validation: {cv} folds) ===") # Loguje nagłówek walidacji krzyżowej.
    start_time = time.time() # Zapisuje czas rozpoczęcia.

    xs, ys = list(param_grid['x']), list(param_grid['y']) # Pobiera listy wartości dla pierwszego (x) i drugiego (y) parametru z param_grid.
    Xp, Yp = np.meshgrid(xs, ys) # Tworzy siatkę punktów (Xp, Yp) z list wartości parametrów.
    Z = np.zeros_like(Xp, dtype=float) # Inicjuje macierz Z do przechowywania wyników.

    for i, xv in enumerate(xs): # Iteruje przez wartości pierwszego parametru (x).
        for j, yv in enumerate(ys): # Iteruje przez wartości drugiego parametru (y).
            params = fixed_params.copy() # Kopiuje stałe parametry.
            params[param_grid['x_name']] = int(xv) if param_grid['x_name'] == 'beta' else xv # Ustawia wartość pierwszego parametru, konwertując na int, jeśli to 'beta'.
            params[param_grid['y_name']] = int(yv) if param_grid['y_name'] == 'beta' else yv # Ustawia wartość drugiego parametru, konwertując na int, jeśli to 'beta'.
            try: # Rozpoczyna blok try-except do obsługi błędów.
                model = model_class(**params) # Tworzy instancję modelu z bieżącymi parametrami.
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric) # Oblicza wyniki walidacji krzyżowej.
                Z[j, i] = scores.mean() # Przypisuje średni wynik do macierzy Z.
            except Exception as e: # Przechwytuje błędy.
                logging.warning(f"Błąd dla parametrów {params}: {e}") # Loguje ostrzeżenie o błędzie.
                Z[j, i] = np.nan # Ustawia wynik na NaN w przypadku błędu.

    elapsed = time.time() - start_time # Oblicza czas wykonania.

    # Szukanie najlepszego i najgorszego punktu
    if np.isnan(Z).all(): # Sprawdza, czy wszystkie wyniki są NaN.
        logging.error("Wszystkie wyniki to NaN — przerwano") # Loguje błąd.
        return None # Zwraca None, jeśli nie ma prawidłowych danych.

    best_idx = np.unravel_index(np.nanargmax(Z), Z.shape) # Znajduje indeks (wiersz, kolumna) największej wartości w Z (ignorując NaN).
    worst_idx = np.unravel_index(np.nanargmin(Z), Z.shape) # Znajduje indeks najmniejszej wartości w Z (ignorując NaN).

    best_x, best_y = Xp[best_idx], Yp[best_idx] # Pobiera wartości parametrów dla najlepszego wyniku.
    best_score = Z[best_idx] # Pobiera najlepszy wynik.
    worst_x, worst_y = Xp[worst_idx], Yp[worst_idx] # Pobiera wartości parametrów dla najgorszego wyniku.
    worst_score = Z[worst_idx] # Pobiera najgorszy wynik.

    # Podsumowanie
    logging.info("=== Podsumowanie wyników ===") # Loguje nagłówek podsumowania.
    logging.info(f"Najlepszy wynik {metric}: {best_score:.4f} przy {param_grid['x_name']}={best_x}, {param_grid['y_name']}={best_y}") # Loguje najlepszy wynik i odpowiadające mu parametry.
    logging.info(f"Najgorszy wynik {metric}: {worst_score:.4f} przy {param_grid['x_name']}={worst_x}, {param_grid['y_name']}={worst_y}") # Loguje najgorszy wynik i odpowiadające mu parametry.
    logging.info(f"Czas zbierania danych: {elapsed:.2f} sekundy") # Loguje czas zbierania danych.
    logging.info("============================") # Loguje separator.

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7)) # Tworzy nową figurę.
    ax = fig.add_subplot(111, projection='3d') # Dodaje oś 3D.
    surf = ax.plot_surface(Xp, Yp, Z, cmap='viridis', edgecolor='k', alpha=0.8) # Tworzy powierzchnię 3D.
    ax.set_xlabel(param_grid['x_name']) # Ustawia etykietę osi X.
    ax.set_ylabel(param_grid['y_name']) # Ustawia etykietę osi Y.
    ax.set_zlabel(metric) # Ustawia etykietę osi Z.
    ax.set_title(title) # Ustawia tytuł wykresu.
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10) # Dodaje pasek koloru.
    ax.scatter(best_x, best_y, best_score, color='r', s=100, label='Najlepszy punkt') # Dodaje czerwony punkt dla najlepszego wyniku.
    ax.scatter(worst_x, worst_y, worst_score, color='black', s=100, label='Najgorszy punkt') # Dodaje czarny punkt dla najgorszego wyniku.
    ax.legend() # Wyświetla legendę.
    save_current_fig(title) # Zapisuje wykres.

    # Widok z góry
    ax.view_init(elev=90, azim=-90) # Zmienia widok na widok z góry.
    save_current_fig(f"{title}_topview") # Zapisuje wykres z widokiem z góry.
    plt.close() # Zamyka figurę.

    return { # Zwraca słownik z wynikami dla najlepszych i najgorszych punktów oraz czasem wykonania.
        param_grid['x_name']: best_x,
        param_grid['y_name']: best_y,
        metric: best_score,
        'worst_' + param_grid['x_name']: worst_x,
        'worst_' + param_grid['y_name']: worst_y,
        'worst_' + metric: worst_score,
        'time_sec': round(elapsed, 2)
    }


# ================================================
# --- beta vs gtol 3D – surface 3D
# ================================================
def plot_beta_gtol_dependency_3d_cv(model_class, betas, gtol_values, fixed_params, # Definiuje funkcję do tworzenia wykresu 3D zależności metryki od parametrów beta i gtol.
                                     X, y, metric='accuracy', title="beta_vs_gtol_cv", cv=CV_FOLDS):
    logging.info(f"Rozpoczynam generowanie wykresu dla modelu: {model_class.__name__}") # Loguje nazwę modelu.
    logging.info(f"=== {model_class.__name__} (Cross-Validation: {cv} folds) ===") # Loguje nagłówek walidacji krzyżowej.

    log_gtol_values = np.log10(gtol_values) # Przekształca wartości gtol na skalę logarytmiczną (log10), co jest często użyteczne dla parametrów o dużym zakresie.
    B, G_log = np.meshgrid(betas, log_gtol_values) # Tworzy siatkę punktów (B, G_log) dla beta i logarytmicznych wartości gtol.
    Z = np.zeros_like(B, dtype=float) # Inicjuje macierz Z do przechowywania wyników.

    start_time = time.time() # Zapisuje czas rozpoczęcia.
    for i, b in enumerate(betas): # Iteruje przez wartości beta.
        for j, g in enumerate(gtol_values): # Iteruje przez wartości gtol.
            params = fixed_params.copy() # Kopiuje stałe parametry.
            params['beta'] = int(b) # Ustawia aktualną wartość beta (konwertuje na int).
            params['gtol'] = float(g) # Ustawia aktualną wartość gtol (konwertuje na float).
            model = model_class(**params) # Tworzy instancję modelu.
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric) # Oblicza wyniki walidacji krzyżowej.
            Z[j, i] = scores.mean() # Przypisuje średni wynik do macierzy Z.

    elapsed = time.time() - start_time # Oblicza czas wykonania.

    # Najlepszy punkt
    idx_best = np.unravel_index(np.argmax(Z), Z.shape) # Znajduje indeks najlepszego wyniku w Z.
    best_beta, best_gtol = B[idx_best], gtol_values[idx_best[0]] # Pobiera wartości beta i gtol dla najlepszego wyniku.
    best_score = Z[idx_best] # Pobiera najlepszy wynik.
    logging.info(f"Najlepszy {metric}={best_score:.4f} przy beta={best_beta}, gtol={best_gtol}") # Loguje szczegóły najlepszego wyniku.
    
    # Najgorszy punkt
    idx_worst = np.unravel_index(np.argmin(Z), Z.shape) # Znajduje indeks najgorszego wyniku w Z.
    worst_beta, worst_gtol = B[idx_worst], gtol_values[idx_worst[0]] # Pobiera wartości beta i gtol dla najgorszego wyniku.
    worst_score = Z[idx_worst] # Pobiera najgorszy wynik.
    logging.info(f"Najgorszy {metric}={worst_score:.4f} przy beta={worst_beta}, gtol={worst_gtol}") # Loguje szczegóły najgorszego wyniku.
    # Wykres 3D
    fig = plt.figure(figsize=(10, 7)) # Tworzy nową figurę.
    ax = fig.add_subplot(111, projection='3d') # Dodaje oś 3D.
    surf = ax.plot_surface(B, G_log, Z, cmap='viridis', edgecolor='k', alpha=0.8) # Tworzy powierzchnię 3D, używając logarytmicznych wartości gtol na osi Y.

    # Dodanie najlepszego punktu (czerwona kula)
    ax.scatter(best_beta, np.log10(best_gtol), best_score, color='red', s=100, label='Najlepszy punkt') # Dodaje czerwony punkt dla najlepszego wyniku na wykresie.

    # Dodanie najgorszego punktu (czarna kula)
    ax.scatter(worst_beta, np.log10(worst_gtol), worst_score, color='black', s=100, label='Najgorszy punkt') # Dodaje czarny punkt dla najgorszego wyniku.

    ax.set_xlabel('beta') # Ustawia etykietę osi X.
    ax.set_ylabel('log10(gtol)') # Ustawia etykietę osi Y (ze względu na skalę logarytmiczną).
    ax.set_zlabel(metric) # Ustawia etykietę osi Z.
    ax.set_title(title) # Ustawia tytuł wykresu.
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10) # Dodaje pasek koloru.
    ax.legend() # Wyświetla legendę.
    save_current_fig(title) # Zapisuje wykres.

    # Widok z góry
    ax.view_init(elev=90, azim=-90) # Zmienia widok na widok z góry.
    save_current_fig(f"{title}_topview") # Zapisuje wykres z widokiem z góry.
    plt.close() # Zamyka figurę.

    return { # Zwraca słownik z wynikami dla najlepszych i najgorszych punktów oraz czasem wykonania.
        'beta': best_beta,
        'gtol': best_gtol,
        metric: best_score,
        'worst_beta': worst_beta,
        'worst_gtol': worst_gtol,
        f'worst_{metric}': worst_score,
        'time_sec': elapsed
    }


# === URUCHOMIENIE ===

models = {'GLVQ': GlvqModel} # Definiuje słownik mapujący nazwy modeli na ich klasy.

# === Wywołanie dla Iris ===
# ================================================
# === SEKCJA URUCHOMIENIA ===
# ================================================
# Zakresy parametrów
max_iters = [5, 10]#, 25, 50, 75, 100, 125, 150, 175, 200, 225] # Definiuje listę wartości dla parametru max_iter do przetestowania.
random_states = [0, 10, 25]#, 50, 75, 100] # Definiuje listę wartości dla parametru random_state do przetestowania.


# Wywołanie funkcji
for name, cls in models.items(): # Iteruje przez każdy model zdefiniowany w słowniku 'models'.
    res1 = plot_maxiter_randomstate_dependency_3d_cv( # Wywołuje funkcję do tworzenia wykresu 3D zależności od max_iter i random_state.
        model_class=cls, # Klasa modelu (np. GlvqModel).
        max_iters=max_iters, # Lista wartości max_iter do testowania.
        random_states=random_states, # Lista wartości random_state do testowania.
        fixed_params=BASE_PARAMS, # Stałe parametry dla modelu.
        X=X_scaled, # Skalowane dane wejściowe.
        y=y, # Etykiety klas.
        metric='accuracy', # Metryka oceny (dokładność).
        title=f"{name}_maxIter_vs_randomState", # Tytuł wykresu.
        cv=CV_FOLDS # Liczba fałd do walidacji krzyżowej.
    )

reset_parameters() # Resetuje globalne parametry do ich początkowych wartości.

# CV trening i ewaluacja
for name, cls in models.items(): # Iteruje przez każdy model.
    train_and_evaluate_cv(cls, name, X_scaled, y, cv=CV_FOLDS, params=BASE_PARAMS) # Trenuje i ocenia model za pomocą walidacji krzyżowej z domyślnymi parametrami.

reset_parameters() # Resetuje globalne parametry.

# CV siatka beta vs prototypes_per_class
for name, cls in models.items(): # Iteruje przez każdy model.
    res = plot_param_dependency_3d_cv( # Wywołuje ogólną funkcję do tworzenia wykresu 3D zależności od 'beta' i 'prototypes_per_class'.
        cls, {'x_name': 'beta', 'y_name': 'prototypes_per_class', 'x': range(2, 5), 'y': range(2, 5)}, # param_grid definiuje zakresy i nazwy parametrów.
        BASE_PARAMS, X_scaled, y, 'accuracy', f"{name}_beta_vs_prototypes_cv", cv=CV_FOLDS # Pozostałe argumenty funkcji.
    )
    logging.info(f"Wyniki {name} beta_vs_prototypes CV: {res}") # Loguje wyniki tej analizy.

reset_parameters() # Resetuje globalne parametry.

# CV beta vs gtol
betas = list(range(2, 4)) # Definiuje listę wartości dla parametru beta.
gtol_vals = [1e-3, 1e-2, 1e-1]#[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] # Definiuje listę wartości dla parametru gtol.
for name, cls in models.items(): # Iteruje przez każdy model.
    res = plot_beta_gtol_dependency_3d_cv( # Wywołuje funkcję do tworzenia wykresu 3D zależności od 'beta' i 'gtol'.
        cls, betas, gtol_vals, BASE_PARAMS, X_scaled, y, 'accuracy', f"{name}_beta_vs_gtol_cv", cv=CV_FOLDS # Pozostałe argumenty funkcji.
    )
    logging.info(f"Wyniki {name} beta_vs_gtol CV: {res}") # Loguje wyniki tej analizy.