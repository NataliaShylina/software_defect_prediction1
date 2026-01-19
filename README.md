# Predykcja defektów oprogramowania na podstawie danych JM1 i Kitchenham

## Skład grupy projektowej i planowany podział zadań
- Nataliia Shylina

## Opis planowanych prac
Celem projektu jest stworzenie modelu predykcji defektów oprogramowania na podstawie cech statycznych kodu (JM1) oraz cech projektowych (Kitchenham). Planowane etapy realizacji obejmują:

1. **Pozyskanie i przygotowanie danych**  
   - Wczytanie zbiorów JM1 (CSV) i Kitchenham (ARFF),
   - Konwersja atrybutów, obsługa braków danych i typów nominalnych,
   - Normalizacja i standaryzacja cech,
   - Utworzenie zmiennej binarnej defect_flag na podstawie kolumny defects.

2. **Eksploracyjna analiza danych (EDA)**  
   - Analiza rozkładów cech i klas,
   - Identyfikacja korelacji i outlierów,
   - Wizualizacje: histogramy, heatmapy korelacji, wykresy zależności.

3. **Budowa modeli predykcyjnych**  
    - Trening modeli klasyfikacyjnych i/lub regresyjnych,
    - Strojenie hiperparametrów (grid search / cross-validation).

4. **Ewaluacja i porównanie modeli**  
    - Ocena jakości predykcji za pomocą odpowiednich miar,
    - Analiza stabilności i interpretowalności wyników. 

5. **Wnioski i rekomendacje**  
    - Wskazanie najlepszych metod,
    - Omówienie przydatności w praktyce inżynierskiej.
   
## Opis problemu do rozwiązania
Celem pracy jest opracowanie i empiryczna ocena modeli predykcji defektów oprogramowania (Software Defect Prediction – SDP), które pozwalają identyfikować moduły lub projekty najbardziej narażone na występowanie błędów jeszcze przed fazą testów lub wdrożenia.

W praktyce inżynierii oprogramowania zasoby testowe są ograniczone, dlatego kluczowe jest wskazanie fragmentów kodu, które mają najwyższe ryzyko defektów. 
    
### Problem polega na:
- dużej liczbie metryk kodu (np. LOC, złożoność cyklomatyczna, miary Halsteada),
- silnej nierównowadze klas (mało modułów z defektami vs. wiele bez defektów),
- heterogeniczności danych projektowych.

Zadaniem jest zbudowanie modeli, które na podstawie metryk statycznych i/lub cech projektowych przewidzą:

- **JM1**: Problem klasyfikacji binarnej – określenie, czy dana jednostka kodu zawiera defekt (`defect_flag = 0/1`) na podstawie cech kodu źródłowego takich jak: liczba linii (`loc`), złożoność cyklomatyczna (`v(g)`), liczba operacji (`total_Op`) itp.  
- **Kitchenham**: Problem regresji lub klasyfikacji – przewidywanie wartości takich jak wysiłek projektowy (`Actual.effort`) lub klasyfikacja projektów pod względem ryzyka defektów na podstawie cech projektu (`Project.type`, `Client.code`, `Adjusted.function.points`, itp.).  

W obu przypadkach celem jest prognozowanie defektów, co pozwala wspierać proces zarządzania jakością oprogramowania.

## Lista metod planowanych do zastosowania

### Klasyfikacja (defekt / brak defektu):
- Regresja logistyczna
- Drzewa decyzyjne
- Random Forest
- Gradient Boosting / XGBoost
- k-NN
- SVM (Support Vector Machines)

### Regresja (liczba defektów – opcjonalnie):
- Regresja liniowa / Ridge / Lasso
- Random Forest Regressor
- Gradient Boosting Regressor 

### Przetwarzanie danych:
- Normalizacja (MinMax / StandardScaler)
- Kodowanie zmiennych nominalnych (One-Hot / LabelEncoder)
- Redukcja wymiaru (PCA – opcjonalnie)

## Wskazanie zbioru danych do treningu i testowania rozwiązania
- **JM1 (jm1.csv)**: 10885 wierszy + 1 nagłówek, cechy statyczne kodu i liczba defektów (`defects`)  
  - Zmienna celu: `defect_flag` (binarna)  
  - Dane podzielone na:  
    - `X_train` / `y_train` – 80% danych, opcjonalnie oversampling SMOTE  
    - `X_test` / `y_test` – 20% danych  

- **Kitchenham (kitchenham.arff)**: dane projektowe (projekty software’owe, różne cechy projektu)  
  - Format: ARFF → wczytywany ręcznie (Python)
  - Dane projektowe: czas trwania, wysiłek, FP, typ projektu, metoda estymacji
  - Zmienna celu: np. `Actual.effort` (regresja) lub wprowadzenie binarnej klasy ryzyka defektu  (jakości)
  - Cross-validation (k=5 lub k=10) 
  - Dane podzielone analogicznie na zbiór treningowy i testowy  

## Opis miar użytych do oceny jakości rozwiązania

### Dla klasyfikacji binarnej (JM1)
- **Accuracy** – ogólna trafność
- **Precision** – dokładność przewidywania defektów (`tp / (tp + fp)`)  
- **Recall (PD)** – odsetek prawdziwych defektów poprawnie wykrytych (`tp / (tp + fn)`)  
- **ROC-AUC** – pole pod krzywą ROC, ocena jakości predykcji prawdopodobieństwa  
- **F1-score** – kompromis precision/recall
- **Confusion Matrix** – analiza FP / FN

Szczególnie ważne: Recall i F1, bo zależy na wykrywaniu defektów.

### Dla regresji (Kitchenham, jeśli przewidywana zmienna ciągła)
- **MAE (Mean Absolute Error)** – średni błąd bezwzględny  
- **RMSE (Root Mean Squared Error)** – pierwiastek z MSE  
- **R²** – współczynnik determinacji - dopasowanie modelu
