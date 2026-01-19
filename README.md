# Predykcja defektów oprogramowania na podstawie danych JM1 i Kitchenham

## Skład grupy projektowej i planowany podział zadań
- Nataliia Shylina

## Opis planowanych prac
Celem projektu jest stworzenie modelu predykcji defektów oprogramowania na podstawie cech statycznych kodu (JM1) oraz cech projektowych (Kitchenham). Praca obejmuje:

1. **Eksplorację danych (EDA)**  
   - Analiza rozkładów cech, braków danych  
   - Korelacje między zmiennymi  
   - Wizualizacje: histogramy, scatterploty, boxploty  

2. **Przygotowanie danych**  
   - Normalizacja zmiennych numerycznych  
   - Konwersja zmiennych jakościowych na liczbową reprezentację (One-hot encoding)  
   - Stworzenie binarnej kolumny `defect_flag`  

3. **Podział danych**  
   - Wydzielenie zbioru treningowego i testowego (np. 80/20)  
   - Ewentualnie walidacja krzyżowa  

4. **Zastosowanie metod uczenia maszynowego**  
   - Klasyfikacja binarna dla JM1  
   - Regresja lub klasyfikacja dla Kitchenham (w zależności od zmiennej celu)  

5. **Ewaluacja modelu**  
   - Miary jakości: Accuracy, Precision, Recall, PD, PF, ROC-AUC  
   - Analiza macierzy pomyłek  

6. **Raportowanie wyników**  
   - Wizualizacja wyników  
   - Interpretacja istotności cech  
   - Analiza błędów

## Opis problemu do rozwiązania
- **JM1**: Problem klasyfikacji binarnej – określenie, czy dana jednostka kodu zawiera defekt (`defect_flag = 0/1`) na podstawie cech kodu źródłowego takich jak: liczba linii (`loc`), złożoność cyklomatyczna (`v(g)`), liczba operacji (`total_Op`) itp.  
- **Kitchenham**: Problem regresji lub klasyfikacji – przewidywanie wartości takich jak wysiłek projektowy (`Actual.effort`) lub klasyfikacja projektów pod względem ryzyka defektów na podstawie cech projektu (`Project.type`, `Client.code`, `Adjusted.function.points`, itp.).  

W obu przypadkach celem jest prognozowanie defektów, co pozwala wspierać proces zarządzania jakością oprogramowania.

## Lista metod planowanych do zastosowania

### Wstępna analiza i przetwarzanie danych
- Normalizacja zmiennych numerycznych (`StandardScaler`)  
- Kodowanie zmiennych kategorycznych (`One-hot encoding`, `Label encoding`)  
- Oversampling danych niezbalansowanych (`SMOTE`)  

### Algorytmy klasyfikacji
- Random Forest Classifier  
- Gradient Boosting (XGBoost / LightGBM)  
- Logistic Regression  
- Support Vector Machine (SVM)  

### Metody ewaluacji i walidacji
- Podział na zbiór treningowy i testowy (`train/test split`)  
- Walidacja krzyżowa (`k-fold CV`)  
- Analiza macierzy pomyłek i krzywych ROC  

### Analiza istotności cech
- Feature importance z Random Forest  
- Analiza korelacji między cechami  

## Wskazanie zbioru danych do treningu i testowania rozwiązania
- **JM1 (jm1.csv)**: 10885 wierszy + 1 nagłówek, cechy statyczne kodu i liczba defektów (`defects`)  
  - Zmienna celu: `defect_flag` (binarna)  
  - Dane podzielone na:  
    - `X_train` / `y_train` – 80% danych, opcjonalnie oversampling SMOTE  
    - `X_test` / `y_test` – 20% danych  

- **Kitchenham (kitchenham.arff)**: dane projektowe (projekty software’owe, różne cechy projektu)  
  - Zmienna celu: np. `Actual.effort` (regresja) lub wprowadzenie binarnej klasy ryzyka defektu  
  - Dane podzielone analogicznie na zbiór treningowy i testowy  

## Opis miar użytych do oceny jakości rozwiązania

### Dla klasyfikacji binarnej (JM1)
- **Accuracy** – dokładność klasyfikacji  
- **Precision** – dokładność przewidywania defektów (`tp / (tp + fp)`)  
- **Recall (PD)** – odsetek prawdziwych defektów poprawnie wykrytych (`tp / (tp + fn)`)  
- **PF** – odsetek fałszywych alarmów (`fp / (fp + tn)`)  
- **ROC-AUC** – pole pod krzywą ROC, ocena jakości predykcji prawdopodobieństwa  

### Dla regresji (Kitchenham, jeśli przewidywana zmienna ciągła)
- **MAE (Mean Absolute Error)** – średni błąd bezwzględny  
- **MSE (Mean Squared Error)** – średni błąd kwadratowy  
- **RMSE (Root Mean Squared Error)** – pierwiastek z MSE  
- **R²** – współczynnik determinacji  

### Dodatkowo
- Analiza macierzy pomyłek i wykresy ROC/Precision-Recall dla interpretowalności modelu  
- Porównanie różnych algorytmów klasyfikacji pod kątem powyższych miar
