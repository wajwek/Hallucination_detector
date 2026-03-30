# hd-dataset-normalizer

Cel: automatycznie rozpoznaj format datasetu (szczególnie HaluEval) i przekonwertuj go do kanonicznego schematu narzędzia Hallucination Detector.

## Co robić

1) Poproś użytkownika o ścieżkę pliku wejściowego datasetu.

2) Jeśli dataset wygląda na HaluEval (kolumny typu `knowledge/question/right_answer/hallucinated_answer` albo podobne), uruchom konwersję:

```powershell
.\.venv\Scripts\python.exe -m hd.cli prepare-halueval --input-path <INPUT> --output-path datasets\prepared\halueval_prepared.jsonl --task auto
```

3) Następnie uruchom detekcję na pliku wynikowym (na start SGI+DC+CAI w trybie hard):

```powershell
.\.venv\Scripts\python.exe -m hd.cli run --dataset datasets\prepared\halueval_prepared.jsonl --enable-sgi true --enable-dc true --enable-cai true --agg hard --run-prefix exp
```

4) Wydrukuj użytkownikowi:
- gdzie jest plik przygotowany
- gdzie zapisano run (`artifacts/runs/<run_id>`)
- podstawowe metryki (accuracy/f1/roc_auc jeśli były label)

## Uwaga
Jeśli input nie jest HaluEval, a to tylko „inne nazwy kolumn”, to nie konwertuj treści, tylko powiedz użytkownikowi jak ustawić mapowanie kolumn w GUI (Dataset/Config).
