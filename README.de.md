[English](README.md) | **Deutsch**

# ML Exercises — Python- & Streamlit-Version

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-testing-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **19 Machine-Learning-Notebooks, portiert in saubere Python-Module und eine interaktive Streamlit-App mit 19 Seiten: Decision Tree → Random Forest → XGBoost auf einem Fleischpreis-Datensatz.**

Dieses Repository ist die Python-Version von [ml_exercises](https://github.com/Martin-Frei/ml_exercises). Aus jedem Jupyter-Notebook dort wurde:

1. **ein eigenständiges Python-Modul** in `classification/` oder `regression/`, mit einem Docstring, der jeden Schritt erklärt, und
2. **eine Streamlit-Seite** in `app/pages/`, auf der man die Daten erkunden, die Modellmetriken ansehen und mit Schiebereglern eigene Vorhersagen machen kann.

Die ursprünglichen Notebooks bleiben die Dokumentation des Lernwegs. Dieses Repository ist die lauffähige, interaktive Version.

> Hinweis: Code, Kommentare und `REPORT.md` / `TESTING.md` sind auf Englisch.

---

## Inhaltsverzeichnis

1. [Schnellstart](#schnellstart)
2. [Die Streamlit-App](#die-streamlit-app)
3. [Streamlit-Befehlsreferenz](#streamlit-befehlsreferenz)
4. [Seiten der App](#seiten-der-app)
5. [Skripte ohne Streamlit ausführen](#skripte-ohne-streamlit-ausführen)
6. [Projektstruktur](#projektstruktur)
7. [Unterschiede zu den Notebooks](#unterschiede-zu-den-notebooks)
8. [Bekannte Probleme](#bekannte-probleme)
9. [Testen](#testen)
10. [Roadmap](#roadmap)
11. [Danksagung](#danksagung)
12. [Lizenz](#lizenz)

---

## Schnellstart

Drei Schritte vom Klonen bis zur laufenden App:

```bash
# 1. Repository klonen und hineinwechseln
git clone https://github.com/Martin-Frei/ML_Exercise_py.git
cd ML_Exercise_py

# 2. Virtuelle Umgebung anlegen und Pakete installieren
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

# 3. App starten
streamlit run app/app.py
```

Der Browser öffnet sich unter **http://localhost:8501**. Über die Seitenleiste links wechselt man zwischen den 19 Modellseiten.

**Wichtig:** Die App immer aus dem **Projekt-Hauptordner** starten (dem Ordner, der `app/`, `shared/`, `regression/` und `classification/` enthält). Die App fügt diesen Ordner zum Importpfad von Python hinzu. Startet man sie aus einem anderen Ordner, kommt `ModuleNotFoundError: No module named 'shared'`.

---

## Die Streamlit-App

### Was die App ist

Streamlit macht aus einem normalen Python-Skript eine Web-App. Es gibt kein HTML, kein JavaScript und keinen Webserver zu konfigurieren. Jedes Bedienelement (Schieberegler, Auswahlliste, Button) ist eine Zeile Python. Bewegt man einen Regler, führt Streamlit das Skript von oben nach unten neu aus und zeichnet die Seite neu.

Die App besteht aus einer **Startseite** (`app/app.py`) und **19 Modellseiten** (`app/pages/`). Streamlit findet die Seiten automatisch und sortiert sie in der Seitenleiste nach der Zahl am Anfang des Dateinamens: `1_DT_Classification.py` steht oben, `19_XGB_Hyperparameter_Tuning.py` unten. Die Reihenfolge folgt dem Lernweg: Decision Tree → Random Forest → XGBoost.

### Was jede Seite zeigt

Die Seiten sind nach drei Vorlagen aufgebaut.

**Regressionsseiten** (Vorhersage von `price_eur_per_kg`):

| Abschnitt | Inhalt |
|---|---|
| Filter in der Seitenleiste | Fleischart, Bio, Preisbereich, Anzahl angezeigter Zeilen |
| Kennzahlen | Gefilterte Zeilen, mittlerer / minimaler / maximaler Preis |
| Datenansicht | Die gefilterten Daten als Tabelle |
| Modellgüte | MAE, RMSE und R² |
| Tatsächlich vs. vorhergesagt | Streudiagramm echter gegen vorhergesagte Preise |
| Feature Importance | Auf welche Merkmale sich das Modell stützt |
| Eigenen Preis vorhersagen | Schieberegler für jedes Merkmal, Button liefert einen Preis |

**Klassifikationsseiten** (Vorhersage von `meat_type`):

| Abschnitt | Inhalt |
|---|---|
| Filter in der Seitenleiste | Fleischart, Bio, Anzahl angezeigter Zeilen |
| Kennzahlen | Gefilterte Zeilen, Anzahl Klassen, Klassenverteilung |
| Datenansicht | Die gefilterten Daten als Tabelle |
| Modellgüte | Accuracy, Precision, Recall, F1 |
| Klassifikationsbericht | Metriken pro Klasse als Tabelle |
| Konfusionsmatrix | Heatmap vorhergesagter gegen echte Klassen |
| Feature Importance | Auf welche Merkmale sich das Modell stützt |
| Fleischart vorhersagen | Schieberegler für jedes Merkmal, Button liefert die Klasse und die Wahrscheinlichkeit jeder Klasse |

**Sonderseite** (Seite 19, Hyperparameter-Tuning): ein Warnhinweis, die Tabelle der besten Hyperparameter, danach die Regressionsabschnitte.

### Zwei Arten von Seiten: live trainiert oder von der Festplatte geladen

| Art | Seiten | Was passiert |
|---|---|---|
| **Beim ersten Aufruf trainiert** | 17 Seiten | Das Modell wird beim ersten Öffnen der Seite trainiert und dann im Arbeitsspeicher gehalten (`@st.cache_resource`). Der erste Aufruf dauert ein paar Sekunden, jeder weitere geht sofort. |
| **Aus gespeichertem Modell geladen** | 6 (RF GridSearch), 19 (XGB Hyperparameter Tuning) | Das Training dauert für einen Seitenaufruf zu lange, daher lädt die Seite eine `.joblib`-Datei aus `models/`. Fehlt die Datei, zeigt die Seite einen Fehler mit dem Befehl zum Erzeugen und hält an. |

So werden die gespeicherten Modelle erzeugt:

```bash
python -m regression.random_forest_grid_search --save-model
python -m regression.xgb_hyperparameter_tuning --save-model --no-plots
```

`--no-plots` verwenden. Ohne diese Option blockiert jedes Diagrammfenster das Skript, bis man es schließt.

### Caching: wann wird ein Modell neu trainiert?

Die trainierten Modelle liegen im Arbeitsspeicher des laufenden Streamlit-Prozesses. Neu trainiert wird, wenn man:

- die App **neu startet** (`Strg+C`, dann wieder `streamlit run app/app.py`),
- in der App **den Cache leert** (Taste `C` oder Menü ⋮ oben rechts → *Clear cache*), oder
- **den Code** der gecachten Trainingsfunktion **ändert**. Streamlit erkennt die Änderung und trainiert neu.

Ändert man die CSV-Dateien in `data/`, muss man den Cache leeren oder neu starten. Sonst zeigen die Seiten weiter das alte Modell.

### Konfiguration: `.streamlit/config.toml`

```toml
[server]
headless = false     # false = Browser beim Start automatisch öffnen

[theme]
base = "light"       # helles Design; "dark" ist die Alternative
```

Streamlit liest diese Datei beim Start. Jede Einstellung kann auch auf der Kommandozeile angegeben werden (siehe nächster Abschnitt). Die Kommandozeile hat Vorrang vor der Datei.

---

## Streamlit-Befehlsreferenz

Alle Befehle werden im Projekt-Hauptordner mit aktivierter virtueller Umgebung ausgeführt.

### Starten und Beenden

| Befehl | Wirkung |
|---|---|
| `streamlit run app/app.py` | Startet die App unter http://localhost:8501 und öffnet den Browser |
| `python -m streamlit run app/app.py` | Dasselbe, nutzt aber das Streamlit des aktiven Pythons. Hilft, wenn `streamlit` „nicht erkannt" wird oder die falsche Version startet |
| `Strg+C` (im Terminal) | Beendet die App |

### Nützliche Optionen für `streamlit run`

| Befehl | Wirkung |
|---|---|
| `streamlit run app/app.py --server.port 8502` | Verwendet Port 8502 statt 8501. Nötig, wenn 8501 schon belegt ist, z. B. durch eine zweite App |
| `streamlit run app/app.py --server.headless true` | Startet, ohne den Browser zu öffnen. Praktisch auf einem Server oder wenn der Tab schon offen ist |
| `streamlit run app/app.py --server.runOnSave true` | Lädt die Seite bei jedem Speichern einer `.py`-Datei automatisch neu. Praktisch beim Bearbeiten einer Seite |
| `streamlit run app/app.py --theme.base dark` | Startet im dunklen Design und überschreibt `config.toml` |
| `streamlit run app/app.py --logger.level debug` | Gibt ausführliche Logs im Terminal aus. Hilft, wenn eine Seite ohne klare Fehlermeldung abbricht |
| `streamlit run app/app.py --browser.gatherUsageStats false` | Schaltet Streamlits anonyme Nutzungsstatistik ab |

Optionen lassen sich kombinieren:

```bash
streamlit run app/app.py --server.port 8502 --server.runOnSave true
```

### Weitere Streamlit-Befehle

| Befehl | Wirkung |
|---|---|
| `streamlit --version` | Zeigt die installierte Streamlit-Version (sollte zu `requirements.txt` passen) |
| `streamlit hello` | Startet Streamlits eingebaute Demo-App. Schneller Test, ob die Installation funktioniert, unabhängig von diesem Projekt |
| `streamlit config show` | Gibt alle Konfigurationsoptionen mit ihren aktuellen Werten aus, inklusive der Werte aus `config.toml` |
| `streamlit cache clear` | Leert Streamlits Cache auf der Festplatte. Diese App hält ihre Modelle **im Arbeitsspeicher**, für ein Neutraining also `C` in der App drücken oder neu starten |
| `streamlit docs` | Öffnet die Streamlit-Dokumentation im Browser |
| `streamlit help` | Listet alle Befehle auf |

### In der laufenden App

| Taste / Menü | Wirkung |
|---|---|
| `R` | Führt die aktuelle Seite neu aus |
| `C` | Leert den Cache, Modelle werden beim nächsten Lauf neu trainiert |
| Menü ⋮ (oben rechts) → *Rerun* / *Clear cache* | Wie `R` / `C` |
| Menü ⋮ → *Settings* | Design wechseln, „Run on save" einschalten |
| *Always rerun* (erscheint nach dem Speichern einer Datei) | Führt die Seite ab jetzt nach jedem Speichern automatisch neu aus |

---

## Seiten der App

| # | Seite | Aufgabe | Modell | Python-Modul | Art |
|---|---|---|---|---|---|
| 1 | DT Classification | Klassifikation | Decision Tree | `classification/decision_tree_classification_meat_type.py` | live trainiert |
| 2 | DT Regression | Regression | Decision Tree | `regression/decision_tree_regression_price_per_kilo.py` | live trainiert |
| 3 | RF Classification | Klassifikation | Random Forest | `classification/random_forest_classification.py` | live trainiert |
| 4 | RF Target Encoding | Klassifikation | Random Forest | `classification/random_forest_target_encoding.py` | live trainiert |
| 5 | RF Regression | Regression | Random Forest | `regression/random_forest_regression.py` | live trainiert |
| 6 | RF GridSearch | Regression | Random Forest | `regression/random_forest_grid_search.py` | **gespeichertes Modell** |
| 7 | RF Outlier Imputation | Regression | Random Forest | `regression/random_forest_regression_outlier_imputation.py` | live trainiert |
| 8 | RF Combined | Regression | Random Forest | `regression/random_forest_combined_dataset.py` | live trainiert |
| 9 | RF Log Price | Regression | Random Forest | `regression/random_forest_log_price.py` | live trainiert |
| 10 | XGB Classification | Klassifikation | XGBoost | `classification/xgb_classification.py` | live trainiert |
| 11 | XGB TE Mean | Klassifikation | XGBoost | `classification/xgb_classification_3A_target_encoding_mean.py` | live trainiert |
| 12 | XGB TE Per Class | Klassifikation | XGBoost | `classification/xgb_classification_3B_target_encoding_per_class.py` | live trainiert |
| 13 | XGB Imputation Clip | Klassifikation | XGBoost | `classification/xgb_classification_imputation_clip.py` | live trainiert |
| 14 | XGB Regression | Regression | XGBoost | `regression/xgb_regression.py` | live trainiert |
| 15 | XGB Regression Clip Fill | Regression | XGBoost | `regression/xgb_regression_clip_fill.py` | live trainiert |
| 16 | XGB OHE | Regression | XGBoost | `regression/xgb_regression_one_hot_encoding.py` | live trainiert |
| 17 | XGB Log OHE | Regression | XGBoost | `regression/xgb_regression_log_one_hot_encoding.py` | live trainiert |
| 18 | XGB Regression Combined | Regression | XGBoost | `regression/xgb_regression_combined.py` | live trainiert |
| 19 | XGB Hyperparameter Tuning | Regression | XGBoost | `regression/xgb_hyperparameter_tuning.py` | **gespeichertes Modell** |

Jedes Modul heißt wie das ursprüngliche Notebook, das passende `.ipynb` findet man also immer in [ml_exercises](https://github.com/Martin-Frei/ml_exercises).

---

## Skripte ohne Streamlit ausführen

Jedes Modul läuft auch eigenständig und gibt dieselben Ergebnisse aus wie das Notebook. Dafür die **Modul-Schreibweise** verwenden (`-m`, Punkte statt Schrägstriche, ohne `.py`):

```bash
python -m classification.xgb_classification
python -m regression.xgb_regression_combined
```

| Option | Wirkung | Verfügbar in |
|---|---|---|
| *(keine)* | Führt die ganze Pipeline aus, gibt Metriken aus und öffnet Diagrammfenster | allen Modulen |
| `--no-plots` | Überspringt alle Diagramme. Für schnelle Läufe und lange Skripte | Modulen mit Diagrammen |
| `--save-plots DIR` | Speichert jedes Diagramm als PNG in `DIR`, statt ein Fenster zu öffnen | Modulen mit Diagrammen |
| `--save-model` | Speichert das trainierte Modell in `models/`, damit die Streamlit-Seite es laden kann | `random_forest_grid_search`, `xgb_hyperparameter_tuning` |
| `--help` | Listet die Optionen eines Moduls auf | allen Modulen mit Optionen |

Beispiele:

```bash
python -m regression.xgb_regression --no-plots
python -m regression.xgb_regression --save-plots plots
python -m regression.xgb_hyperparameter_tuning --save-model --no-plots
```

Warum die Modul-Schreibweise? Die Module importieren aus `shared/` (Daten laden, Metriken, Modelle speichern). `python -m` führt sie als Teil des Projektpakets aus, dadurch funktionieren diese Importe. `python regression/xgb_regression.py` würde mit `ModuleNotFoundError` abbrechen.

---

## Projektstruktur

```
ML_Exercise_py/
│
├── app/
│   ├── app.py                      # Streamlit-Startseite (Einstiegspunkt)
│   └── pages/                      # 19 Modellseiten, 1_ ... 19_, von Streamlit automatisch gefunden
│
├── classification/                 # 7 Module, Zielvariable = meat_type
├── regression/                     # 12 Module, Zielvariable = price_eur_per_kg
│
├── shared/
│   ├── data_loading.py             # Lädt die beiden CSV-Dateien
│   └── modeling.py                 # Metriken, Imputation, Modelle speichern/laden
│
├── data/
│   ├── meat_price_dataset.csv      # 1.200 saubere Zeilen
│   └── meat_price_100_stress_test.csv  # 100 Out-of-Distribution-Zeilen
│
├── models/                         # Gespeicherte Modelle für die Seiten 6 und 19 (.joblib)
├── .streamlit/config.toml          # Streamlit-Einstellungen (Browser, Design)
│
├── requirements.txt
├── REPORT.md                       # Portierungsprotokoll: Ergebnisse und Bugfixes pro Notebook
├── TESTING.md                      # Test-Checkliste mit erwarteten Werten
├── README.md                       # Englisch
└── README.de.md                    # Deutsch
```

Der Datensatz selbst ist ausführlich im [ml_exercises-README](https://github.com/Martin-Frei/ml_exercises/blob/main/README.de.md) beschrieben.

---

## Unterschiede zu den Notebooks

Die Portierung wurde mit Claude Code in fünf Etappen gemacht. Jede Etappe wurde geprüft, bevor die nächste begann. Das vollständige Protokoll steht in [REPORT.md](REPORT.md). Die wichtigsten Änderungen:

| Änderung | Wo | Wirkung |
|---|---|---|
| Stresstest nur auf den **91 Zeilen mit echtem Preis** ausgewertet, nicht auf 100 Zeilen, von denen 9 einen imputierten Preis hatten | XGB-Regressionsmodule | Ehrlichere Stress-Metriken; kleine Abweichungen zu den Notebooks (z. B. XGB-Baseline Stress-R² −0,346 statt −0,359) |
| Leakage in der Imputation entfernt: Fehlende Werte werden nicht mehr mit dem Median der eigenen `meat_type`-Gruppe gefüllt, denn `meat_type` ist die Zielvariable | RF-Klassifikation (Seiten 3, 4) | Stress-Accuracy sinkt (27,5 % → 19,8 %, 25,3 % → 17,6 %). Der niedrigere Wert ist der ehrliche |
| Syntaxfehler behoben (nicht geschlossener String) | RF-Regression | Modul läuft wieder |
| Zellreihenfolge, nicht trainierte Baseline, absolute Dateipfade, deutsche Kommentare behoben | XGB-Hyperparameter-Tuning | Modul läuft wieder |
| Diagramme über `--no-plots` / `--save-plots` steuerbar | alle Module mit Diagrammen | Keine blockierenden Fenster, keine fest eingetragenen Pfade |

XGBoost-Ergebnisse können um etwa 0,5–1 Prozentpunkte von den Notebooks abweichen, weil sich die XGBoost-Version geändert hat.

---

## Bekannte Probleme

1. **Die One-Hot-Encoding-Seiten kodieren nicht one-hot** (Seiten 16 und 17). `pd.get_dummies` bewirkt bei der Integer-Spalte `meat_type` nichts. Die ursprünglichen Notebooks hatten dasselbe Problem, ihre Schlussfolgerung „One-Hot-Encoding zersplittert das stärkste Merkmal" ist durch das Experiment also nicht belegt. Lösung: `meat_type` vor `get_dummies` in eine Kategorie umwandeln.
2. **Die Stresstest-Werte auf Seite 19 sind nicht gültig.** Etwa 73 der 91 Stress-Zeilen liegen im Trainingsset. Das Stress-R² misst daher, wie viel ein Modell auswendig gelernt hat, nicht wie gut es generalisiert: Die flexible Baseline erreicht 0,72, die stark regularisierten getunten Modelle 0,03–0,08. Das gültige Ergebnis von Seite 19 ist die Verbesserung auf dem zurückgehaltenen gemischten Testset (R² 0,868 → 0,895, MAE 1,89 → 1,56 EUR).
3. **Die Laufzeitangabe auf Seite 19 stimmt nicht.** Dort stehen 8–25 Minuten, das Tuning ist tatsächlich in Sekunden fertig.
4. **Feinsuche auf Seite 19:** `learning_rate` enthält `0.001`, damit lernen 250–350 Bäume nichts Brauchbares. Außerdem fehlt `colsample_bylevel` (0,7 in der Grobsuche) und fällt unbemerkt auf 1,0 zurück.
5. **RF Combined (Seite 8)** berechnet die Füllwerte auf den kombinierten Daten, inklusive der Testzeilen. Das ist ein kleines Leck, das beibehalten wurde, um dem Original-Notebook zu entsprechen.
6. **Kleine Stichproben.** Positive Stress-Ergebnisse (Seite 18: R² +0,197) beruhen auf 18–21 Stress-Zeilen. Sie sind aussagekräftig, aber verrauscht.

---

## Testen

Die App ist noch nicht vollständig getestet. [TESTING.md](TESTING.md) enthält:

- eine Prüfung der Umgebung,
- ein Skript, das jedes Modul einmal ausführt,
- eine Checkliste für alle 19 Seiten mit den erwarteten Metrikwerten aus der Portierung und
- die häufigsten Fehler und wie man sie behebt.

---

## Roadmap

- [x] Alle 19 Notebooks in Python-Module portiert
- [x] Streamlit-App mit 19 Seiten
- [x] Seiten in der Reihenfolge des Lernwegs sortiert
- [ ] Test-Checkliste in TESTING.md abarbeiten
- [ ] One-Hot-Encoding auf den Seiten 16 und 17 korrigieren
- [ ] Seite 19: Stresstest nur auf den 18 zurückgehaltenen Stress-Zeilen auswerten, mit wiederholten Splits
- [ ] Seite 19: Laufzeitangabe und Feinsuche korrigieren
- [ ] `LICENSE`-Datei hinzufügen
- [ ] Deployment auf Streamlit Community Cloud

---

## Danksagung

Dieses Projekt entstand im Rahmen von ML-Tutoring-Sessions mit [Adeena](https://github.com/Adeenasamoo), die die Experimente begleitet und die Ergebnisse geprüft hat.

---

## Lizenz

MIT-Lizenz.

**Autor:** Martin Freimuth — [GitHub](https://github.com/Martin-Frei)
