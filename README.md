Analiza i Wizualizacja Modeli LVQ
To repozytorium zawiera kod w Pythonie do analizy i wizualizacji wydajności modeli Uogólnionej Kwantyzacji Wektorowej Uczącej się (GLVQ). Wykorzystuje biblioteki scikit-learn i sklearn-lvq do przeprowadzania walidacji krzyżowej, strojenia hiperparametrów oraz tworzenia wykresów 3D dokładności modelu w różnych kombinacjach parametrów.
Funkcjonalności
Implementacja Modelu GLVQ: Wykorzystuje GlvqModel z biblioteki sklearn-lvq.
Integracja Zbioru Danych Iris: Używa dobrze znanego zbioru danych Iris do zadań klasyfikacji.
Przetwarzanie Wstępne Danych: Implementuje StandardScaler dla solidnego skalowania danych.
Walidacja Krzyżowa: Wykorzystuje cross_val_score i cross_val_predict do wiarygodnej oceny modelu.
Metryki Wydajności: Generuje szczegółowy classification_report i accuracy_score.
Logowanie: Konfiguruje kompleksowy system logowania do śledzenia wykonania i wyników zarówno w konsoli, jak i w pliku (console.log).
Wizualizacja 3D: Generuje interaktywne wykresy powierzchni 3D za pomocą matplotlib, aby zilustrować wpływ:
max_iter vs. random_state na dokładność modelu.
beta vs. prototypes_per_class na dokładność modelu.
beta vs. gtol na dokładność modelu.
Automatyczne Zapisywanie Wykresów: Automatycznie zapisuje wygenerowane wykresy do wyznaczonego katalogu figures w wysokiej rozdzielczości.
Rozpoczęcie Pracy
Wymagania Wstępne
Przed uruchomieniem kodu upewnij się, że masz zainstalowane następujące biblioteki Python:
pip install numpy matplotlib scikit-learn sklearn-lvq


Struktura Projektu
├── main.py             # Główny skrypt do analizy modelu i tworzenia wykresów
├── console.log         # Plik logów dla wykonania skryptu (generowany automatycznie)
└── figures/            # Katalog do zapisywania wygenerowanych wykresów (tworzony automatycznie)
    ├── *.png           # Zapisane wykresy 3D i ich widoki z góry


Uruchamianie Skryptu
Sklonuj repozytorium (jeśli dotyczy):
git clone <adres_url_repozytorium>
cd <nazwa_repozytorium>


Wykonaj główny skrypt:
python main.py


Po wykonaniu skryptu:
Załaduje i przeskaluje zbiór danych Iris.
Przeprowadzi walidację krzyżową i wygeneruje raport klasyfikacji dla modelu GLVQ z domyślnymi parametrami.
Wygeneruje i zapisze wykresy 3D pokazujące zależność między dokładnością a określonymi parami hiperparametrów (max_iter vs. random_state, beta vs. prototypes_per_class, beta vs. gtol).
Zaloguje wszystkie istotne zdarzenia i wyniki do console.log oraz wyświetli je w konsoli.
Konfiguracja
Skrypt zawiera globalne parametry i funkcje ułatwiające dostosowanie:
Logowanie
log_filename: Definiuje nazwę pliku logów (console.log).
Poziom logowania jest ustawiony na INFO.
Katalog Wykresów
figures_dir: Określa katalog, w którym będą zapisywane wykresy (figures).
Globalne Parametry Modelu (BASE_PARAMS)
Możesz modyfikować domyślne parametry dla modelu GLVQ:
BASE_PARAMS = {
    'prototypes_per_class': 5,
    'initial_prototypes': None,
    'max_iter': 1000,
    'gtol': 1e-5,
    'beta': 2,
    'random_state': 42,
    'display': False
}


Liczba Fałd Walidacji Krzyżowej (CV_FOLDS)
Liczbę fałd dla walidacji krzyżowej można dostosować:
CV_FOLDS = 5


Zakresy Hiperparametrów dla Wykresów 3D
Zakresy dla max_iters, random_states, betas i gtol_values w wykresach 3D można dostosować w sekcji === URUCHOMIENIE === skryptu.
Funkcje
save_current_fig(title): Zapisuje aktualnie aktywną figurę matplotlib do katalogu figures.
reset_parameters(): Resetuje globalne BASE_PARAMS i CV_FOLDS do ich początkowych, zdefiniowanych wartości.
train_and_evaluate_cv(model_class, name, X, y, cv=CV_FOLDS, params=None): Trenuje i ocenia model za pomocą walidacji krzyżowej, logując dokładność i raport klasyfikacji.
plot_maxiter_randomstate_dependency_3d_cv(...): Generuje wykres powierzchni 3D pokazujący wpływ max_iter i random_state na dokładność modelu.
plot_param_dependency_3d_cv(...): Uogólniona funkcja do tworzenia wykresu zależności dokładności od dowolnych dwóch określonych parametrów.
plot_beta_gtol_dependency_3d_cv(...): Specyficznie rysuje wpływ beta i gtol na dokładność modelu, z gtol na skali logarytmicznej dla lepszej wizualizacji.
Wynik
Skrypt wygeneruje:
Wyjście konsoli szczegółowo opisujące proces treningu, wyniki walidacji krzyżowej oraz najlepsze/najgorsze kombinacje parametrów znalezione podczas generowania wykresów 3D.
Plik console.log zawierający te same komunikaty logów.
Obrazy PNG w katalogu figures, prezentujące powierzchnie dokładności 3D i ich widoki z góry dla różnych zależności hiperparametrów.
