# software_defect_prediction_clean.py
"""
Poprawiony skrypt do eksploracji datasetu jm1.csv (Software Defect Prediction).
- normalizuje kolumnę 'defects'
- tworzy 'defect_flag' (binarną)
- wypisuje summary, rysuje wykresy i zapisuje przetworzone dane
Uruchom: python software_defect_prediction_clean.py
"""

import os
import sys
import math
import logging
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Opcjonalnie plotly — jeśli instalujesz plotly, wykresy interaktywne zostaną wyświetlone.
try:
    from plotly.offline import iplot
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------- Ustawienia ----------
INPUT_CSV = "jm1.csv"
OUTPUT_PREPROCESSED = "jm1_preprocessed.csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
# ---------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Wczytaj CSV z uwzględnieniem błędów odczytu."""
    if not os.path.exists(path):
        log.error("Plik nie istnieje: %s", path)
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
        log.info("Wczytano plik: %s (shape=%s)", path, df.shape)
        return df
    except Exception as e:
        log.exception("Błąd wczytywania CSV: %s", e)
        raise

def normalize_defects_column(df: pd.DataFrame, col: str = "defects") -> pd.DataFrame:
    """
    Ujednolica kolumnę 'defects':
    - próbuje mapować {'TRUE':1,'FALSE':0} z uwzględnieniem różnych przypadków,
    - jeśli kolumna jest numeryczna, zachowuje ją,
    - tworzy dodatkową kolumnę 'defect_flag' (0/1) - binarną.
    """
    if col not in df.columns:
        log.error("Kolumna '%s' nie istnieje w DataFrame", col)
        raise KeyError(col)

    s = df[col]

    # 1) spróbuj mapowania standardowego (wielkie litery)
    mapped = s.map({'TRUE': 1, 'FALSE': 0})
    if mapped.isna().sum() < len(s):
        df[col + "_mapped_case"] = mapped
        log.info("Część wartości zmapowana przez {'TRUE':'1','FALSE':'0'}: nałożono kolumnę %s", col + "_mapped_case")
    else:
        df[col + "_mapped_case"] = pd.NA

    # 2) spróbuj mapowania małe litery / trim
    if df[col + "_mapped_case"].isna().all():
        try:
            mapped_lower = s.astype(str).str.strip().str.lower().map({'true': 1, 'false': 0})
            if mapped_lower.notna().any():
                df[col + "_mapped_case"] = mapped_lower
                log.info("Zmapowano wartości po lower(): przykładów TRUE/FALSE: %d", mapped_lower.notna().sum())
        except Exception:
            pass

    # 3) jeśli kolumna wygląda numerycznie -> zachowaj wartości numeryczne
    numeric_attempt = None
    try:
        numeric_attempt = pd.to_numeric(s, errors='coerce')
        if numeric_attempt.notna().sum() > 0 and numeric_attempt.notna().sum() != len(s):
            # jeśli część wartości numerycznych, część nie - zachowaj ale wskażmy
            df[col + "_numeric"] = numeric_attempt
            log.info("Część wartości mogła być skonwertowana na numeryczne (kolumna %s)", col + "_numeric")
        elif numeric_attempt.notna().sum() == len(s):
            # wszystkie numeric
            df[col + "_numeric"] = numeric_attempt
            log.info("Cała kolumna traktowana jako numeryczna - zapisana w %s", col + "_numeric")
    except Exception:
        pass

    # 4) Ostateczna decyzja - utwórz defect_flag (0/1)
    # Preferencja: mapped_case -> numeric -> fallback na (non-null -> 1)
    if df.get(col + "_mapped_case") is not None and df[col + "_mapped_case"].notna().any():
        df["defect_flag"] = df[col + "_mapped_case"].fillna(0).astype(int)
    elif df.get(col + "_numeric") is not None and df[col + "_numeric"].notna().any():
        # załóżmy: jeśli numeric > 0 => defect
        df["defect_flag"] = (df[col + "_numeric"] > 0).astype(int)
    else:
        # fallback jeżeli wszystko inne nie zadziała: traktujemy nie-puste jako 1
        df["defect_flag"] = (~s.isna() & (s.astype(str).str.strip() != "")).astype(int)
        log.warning("Użyto fallbacku do utworzenia 'defect_flag' (niezmapowane/nenulowe => 1)")

    # informacja o brakach
    n_nan_original = s.isna().sum()
    n_nan_flag = df["defect_flag"].isna().sum()
    log.info("Original NaNs in '%s': %d ; NaNs in 'defect_flag': %d", col, n_nan_original, n_nan_flag)

    return df

def safe_value_counts(series: pd.Series) -> pd.Series:
    """Zwraca uporządkowane value_counts bez overflow dla dużej liczby unikatów."""
    try:
        vc = series.value_counts(dropna=False)
        return vc
    except OverflowError:
        # fallback: zlicz unikalne wartości iteracyjnie (bez tworzenia dużych tablic)
        from collections import Counter
        c = Counter(series.fillna("__NA__").astype(str))
        vc = pd.Series(dict(c)).sort_values(ascending=False)
        return vc

def plot_and_save_histogram_binary(series: pd.Series, filename: str, title: str = "Distribution"):
    """Rysuje histogram (matplotlib) i zapisuje plik."""
    plt.figure(figsize=(6,4))
    counts = series.value_counts().sort_index()
    ax = counts.plot(kind='bar', rot=0)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Counts")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out)
    plt.close()
    log.info("Zapisano histogram: %s", out)

def try_plotly_hist(series: pd.Series):
    """Opcjonalny wykres plotly jeśli biblioteka dostępna."""
    if not PLOTLY_AVAILABLE:
        log.debug("Plotly niedostępne, pomijam wykres interaktywny.")
        return
    try:
        trace = go.Histogram(x=series, opacity=0.75, marker=dict(color='green'))
        layout = go.Layout(title=f"{series.name} (plotly)")
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
    except Exception as e:
        log.warning("Nie udało się narysować wykresu plotly: %s", e)

def correlation_and_heatmap(df: pd.DataFrame, filename: str = "heatmap.png"):
    """Oblicza korelacje numerycznych kolumn i rysuje heatmapę (seaborn)."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        log.warning("Za mało kolumn numerycznych do korelacji.")
        return
    corr = num_df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.3)
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out)
    plt.close()
    log.info("Zapisano heatmapę korelacji: %s", out)
    return corr

def scatter_if_columns_exist(df: pd.DataFrame, x_col: str, y_col: str, filename: str, title: str):
    """Narysuj scatter plot jeśli kolumny istnieją."""
    if x_col not in df.columns or y_col not in df.columns:
        log.debug("Brakuje kolumn do scatter: %s lub %s", x_col, y_col)
        return
    plt.figure(figsize=(6,4))
    plt.scatter(df[x_col], df[y_col], s=12, alpha=0.6)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out)
    plt.close()
    log.info("Zapisano scatter: %s", out)

def boxplot_if_columns_exist(df: pd.DataFrame, x_col: str, y_col: str, filename: str, title: str):
    """Narysuj box plot (y względem x) jeśli kolumny istnieją."""
    if x_col not in df.columns or y_col not in df.columns:
        log.debug("Brakuje kolumn do boxplot: %s lub %s", x_col, y_col)
        return
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[x_col].astype(str), y=df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, filename)
    plt.savefig(out)
    plt.close()
    log.info("Zapisano boxplot: %s", out)

def main():
    log.info("Start skryptu")

    # 1. Wczytaj dane
    df = load_data(INPUT_CSV)

    # 2. Podstawowe info (używamy DataFrame.info() - nazwa nie koliduje z modułem nltk itp.)
    log.info("Informacje o DataFrame:")
    df.info()   # wypisze do stdout
    log.info("Pierwsze 5 wierszy:")
    log.info("\n%s", df.head().to_string())
    log.info("Ostatnie 5 wierszy:")
    log.info("\n%s", df.tail().to_string())

    # 3. Normalizacja kolumny 'defects' i stworzenie 'defect_flag'
    df = normalize_defects_column(df, col="defects")
    log.info("'defect_flag' distribution:\n%s", safe_value_counts(df["defect_flag"]))

    # 4. Zapisz przetworzone dane
    try:
        df.to_csv(OUTPUT_PREPROCESSED, index=False)
        log.info("Zapisano przetworzone dane do: %s", OUTPUT_PREPROCESSED)
    except Exception as e:
        log.warning("Nie udało się zapisać przetworzonych danych: %s", e)

    # 5. Rysowanie podstawowych wykresów
    # histogram klasy binarnej
    plot_and_save_histogram_binary(df["defect_flag"], "defect_flag_hist.png", title="Defect flag distribution (0/1)")
    # optional plotly
    try_plotly_hist(df["defect_flag"])

    # 6. Korelacje/heatmap
    corr = correlation_and_heatmap(df, filename="heatmap_numeric.png")

    # 7. Scatter i box (jeśli kolumny istnieją)
    # popularne nazwy w datasetach (jm1) to: 'v' (volume), 'b' (bug count), 'v(g)' (cyclomatic complexity), 'loc' etc.
    scatter_if_columns_exist(df, "v", "b", "scatter_v_b.png", "Volume vs Bug Count")
    scatter_if_columns_exist(df, "v(g)", "b", "scatter_vg_b.png", "Cyclomatic Complexity v(g) vs Bug Count")
    boxplot_if_columns_exist(df, "defect_flag", "b", "box_defectflag_b.png", "Defect flag vs Bug Count")

    # 8. Dodatkowe statystyki - percentyle, outliers dla kolumn liczbowych
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]).T
        log.info("Opis statystyczny kolumn numerycznych (wybrane percentyle):\n%s", desc.to_string())
        try:
            desc.to_csv(os.path.join(PLOT_DIR, "numeric_description_percentiles.csv"))
            log.info("Zapisano plik z opisem percentyli.")
        except Exception:
            pass

    log.info("Koniec. Pliki wykresów znajdują się w katalogu: %s", PLOT_DIR)

if __name__ == "__main__":
    main()
