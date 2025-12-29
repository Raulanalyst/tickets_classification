# Clasificación de Tickets

**Propósito**: Conjunto de scripts para limpiar datos de tickets, entrenar un clasificador SVM sobre texto (TF‑IDF) y generar visualizaciones de resultados.

## Archivos principales

- **`modelo_svm.py`** — Entrenamiento y evaluación del clasificador SVM.
- **`clean_data.py`** — Limpieza y filtrado del JSONL original para generar `tickets_cleaned.jsonl`.
- **`grafica_resultados.py`** — Carga de modelo entrenado y generación de 7 gráficas de evaluación.

## Cómo usar (resumen rápido)

1. **Limpiar datos**:
   - Ejecutar: `python clean_data.py`
   - Entrada: `tickets.jsonl`
   - Salida: `tickets_cleaned.jsonl`

2. **Entrenar y evaluar**:
   - Ejecutar: `python modelo_svm.py`
   - Lee `tickets_cleaned.jsonl`, entrena el modelo, evalúa, guarda `ticket_classifier_svm.pkl` y genera gráficas en `graficas_svm`.

3. **Visualizaciones (modo independiente)**:
   - Ejecutar: `python grafica_resultados.py`
   - Carga `ticket_classifier_svm.pkl` y `tickets_cleaned.jsonl`, genera las 7 imágenes en `graficas_svm`.

## Requisitos principales

- Python 3.8+
- Paquetes (ejemplos): `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `tqdm`
- **Nota**: `clean_data.py` usa recursos NLTK (corpora words, stopwords) — ejecutar la descarga de recursos NLTK si es necesario.

---

## `clean_data.py` (`clean_data.py`)

**Propósito**: Leer el archivo JSONL de tickets crudos, filtrar registros inválidos y escribir un JSONL limpio en batches.

**Comportamiento clave**:
- Cuenta líneas y procesa en streaming con `tqdm`.
- Filtra tickets sin `description_plain`, sin `name`, sin `ticket_type_id_name` o con tipos no deseados (`Question`, `Duplicate`, `Undefined`).
- Usa un detector simple `contains_english_word()` que compara palabras contra el vocabulario NLTK `words` excluyendo stopwords.
- Limpia el campo `name` removiendo paréntesis.
- Escribe en batches para eficiencia y muestra un resumen de conteos.

**Resultado**: `tickets_cleaned.jsonl` con registros listos para entrenamiento.

---

## `modelo_svm.py` (`modelo_svm.py`)

**Propósito**: Implementa la clase `TicketClassifierSVM` para entrenar, predecir, evaluar y exportar un SVM lineal sobre features TF‑IDF.

**Componentes principales**:
- `TicketClassifierSVM.__init__`: configura `TfidfVectorizer` (uni+bigramas, min_df, max_df, stopwords en inglés, sublinear TF) y `LinearSVC` con `class_weight='balanced'`.
- `load_jsonl(file_path)`: (método estático) lectura de JSONL en DataFrame (usa `tickets_cleaned.jsonl` en el flujo principal).
- `combine_text(df)`: concatena `name + description` (posible duplicación del `name` para más peso).
- `analyze_distribution(y)`: imprime y guarda distribución de clases (detecta desbalance).
- `train(X_train, y_train)`: codifica etiquetas, calcula pesos de clase con `compute_class_weight`, entrena pipeline (tfidf + svm) y reporta vocabulario y clases.
- `predict`, `predict_with_confidence`: predicción normal y con score aproximado (decision function).
- `evaluate(X_test, y_test)`: métricas globales y por clase, matriz simplificada de confusión, imprime reportes.
- `plot_results(...)`: (genera múltiples gráficas; puede delegar a `grafica_resultados.py` en uso separado).
- `get_feature_importance(top_n)`: extrae términos más influyentes por clase a partir del modelo SVM (peso de coeficientes).
- `save_model(path)`: guarda pipeline, label encoder y distribución en `ticket_classifier_svm.pkl`.

**Flujo del script**:
Carga datos, split estratificado con `train_test_split`, instancia `TicketClassifierSVM`, prepara X/y, entrena, evalúa, extrae features importantes, genera gráficas y guarda el modelo.

---

## `grafica_resultados.py` (`grafica_resultados.py`)

**Propósito**: Clase `TicketClassifierVisualizer` para cargar un modelo guardado y producir una colección de gráficas explicativas y resumen.

**Componentes principales**:
- `load_model(path)`: carga `ticket_classifier_svm.pkl` (espera un dict con `pipeline`, `label_encoder`, `class_distribution`).
- `load_jsonl(file_path)`: lee JSONL y construye DataFrame con las columnas `name`, `description`, `type`.
- `combine_text`, `predict`: utilidades análogas para preparar texto y predecir.
- `plot_results(X_test, y_test, y_pred, output_dir)`: genera y guarda **7 gráficas**:
  1. Distribución de clases (top 15)
  2. F1-Score por categoría (top 15)
  3. Accuracy global (pie + estadísticas)
  4. Matriz de confusión normalizada (top 15 si hay muchas clases)
  5. Comparación Precision/Recall/F1 (top 15)
  6. Support por categoría (top 15)
  7. Resumen de estadísticas (texto con métricas macro/weighted y conteos)

**Salida**: imágenes PNG en `graficas_svm` y resumen impreso en consola.