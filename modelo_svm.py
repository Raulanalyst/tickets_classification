import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings

warnings.filterwarnings('ignore')

class TicketClassifierSVM:

    def __init__(self, max_features=15000, ngram_range=(1, 2), C=1.0):
        """
        Clasificador de tickets usando SVM con TF-IDF
        Optimizado para clases desbalanceadas
        
        Args:
            max_features: Número máximo de features TF-IDF
            ngram_range: Rango de n-gramas (1,2) = uni+bigramas
            C: Parámetro de regularización de SVM
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        
        # Vectorizador TF-IDF optimizado para inglés
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=3,  # Mínimo 3 documentos
            max_df=0.85,  # Máximo 85% de documentos
            sublinear_tf=True,
            strip_accents='ascii',  # Optimizado para inglés
            lowercase=True,
            stop_words='english',  # Elimina stopwords comunes del inglés
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Solo palabras alfabéticas (inglés)
        )
        
        # SVM con balance de clases
        self.svm_model = LinearSVC(
            C=C,
            max_iter=2000,
            random_state=42,
            dual='auto',
            class_weight='balanced',  # CRÍTICO para clases desbalanceadas
            verbose=0
        )
        
        # Pipeline completo
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('svm', self.svm_model)
        ])
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_distribution = None
        
    @staticmethod
    def load_jsonl(file_path):
        """
        Carga datos desde archivo JSONL
        
        Args:
            file_path: Ruta al archivo .jsonl
        
        Returns:
            DataFrame con las columnas: name, description, type
        """
        data = []
        print(f"Cargando datos desde: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    ticket_type = record['ticket_type_id_name']
                    if ticket_type != 'Undefined':
                        data.append({
                            'name': record['name'],
                            'description': record['description_plain'],
                            'type': ticket_type
                        })
                except json.JSONDecodeError as e:
                    print(f"⚠ Error en línea {i+1}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        print(f"✓ Cargados {len(df):,} registros")
        return df
    
    def combine_text(self, df):
        """
        Combina 'name' y 'description' dando más peso al nombre
        """
        name = df['name'].fillna('').astype(str)
        description = df['description'].fillna('').astype(str)
        
        # Se puede duplicar el nombre para darle más importancia
        # name + ' ' + name + ' ' + description
        return name + ' ' + description
    
    def analyze_distribution(self, y):
        """
        Analiza y muestra la distribución de clases
        """
        distribution = pd.Series(y).value_counts().sort_values(ascending=False)
        total = len(y)
        
        print("\n" + "="*70)
        print("DISTRIBUCIÓN DE CLASES")
        print("="*70)
        print(f"{'Categoría':<35} {'Cantidad':>12} {'Porcentaje':>12}")
        print("-"*70)
        
        for category, count in distribution.items():
            percentage = (count / total) * 100
            print(f"{category:<35} {count:>12,} {percentage:>11.2f}%")
        
        print("-"*70)
        print(f"{'TOTAL':<35} {total:>12,} {100.0:>11.2f}%")
        print()
        
        # Identificar clases minoritarias
        min_samples = distribution.min()
        max_samples = distribution.max()
        ratio = max_samples / min_samples
        
        print(f"Desbalance:")
        print(f"  - Clase más frecuente: {max_samples:,} muestras")
        print(f"  - Clase menos frecuente: {min_samples:,} muestras")
        print(f"  - Ratio de desbalance: {ratio:.1f}:1")
        
        self.class_distribution = distribution
        
        return distribution
    
    def train(self, X_train, y_train):
        """
        Entrena el modelo SVM con manejo de clases desbalanceadas
        """
        print("="*70)
        print("INICIANDO ENTRENAMIENTO")
        print("="*70)
        print(f"Muestras de entrenamiento: {len(X_train):,}")
        print(f"Clases únicas: {len(set(y_train))}")
        
        # Analizar distribución
        self.analyze_distribution(y_train)
        
        # Codificar etiquetas
        print("1. Codificando etiquetas...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Calcular pesos de clase
        print("2. Calculando pesos para clases desbalanceadas...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        
        print("   Pesos por clase:")
        for i, (cls, weight) in enumerate(zip(self.label_encoder.classes_, class_weights)):
            count = sum(y_train_encoded == i)
            print(f"     {cls:<35} peso: {weight:6.3f} (n={count:,})")
        
        # Entrenar pipeline
        print("\n3. Vectorizando textos con TF-IDF...")
        print("4. Entrenando modelo SVM (esto puede tomar varios minutos)...")
        
        self.pipeline.fit(X_train, y_train_encoded)
        
        self.is_trained = True
        
        print("\n✓ Entrenamiento completado exitosamente!")
        print(f"  - Vocabulario TF-IDF: {len(self.vectorizer.vocabulary_):,} términos")
        print(f"  - Clases entrenadas: {len(self.label_encoder.classes_)}")
    
    def predict(self, X_test):
        """Predice las categorías de nuevos tickets"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")
        
        y_pred_encoded = self.pipeline.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_with_confidence(self, X_test):
        """
        Predice con scores de decisión (confianza aproximada)
        
        Returns:
            DataFrame con predicción y score por cada muestra
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        y_pred = self.predict(X_test)
        decision_scores = self.pipeline.decision_function(X_test)
        
        # Para cada muestra, obtener el score máximo
        if len(self.label_encoder.classes_) > 2:
            max_scores = np.max(decision_scores, axis=1)
        else:
            max_scores = np.abs(decision_scores)
        
        results = pd.DataFrame({
            'prediction': y_pred,
            'confidence_score': max_scores
        })
        
        return results
    
    def evaluate(self, X_test, y_test, show_per_class=True):
        """
        Evalúa el modelo mostrando métricas adaptadas a clases desbalanceadas
        """
        print("\n" + "="*70)
        print("EVALUACIÓN DEL MODELO")
        print("="*70)
        print(f"Muestras de prueba: {len(X_test):,}\n")
        
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Métricas globales
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print()
        
        # Reporte detallado
        print("Reporte de Clasificación (métricas ponderadas por soporte):")
        print("-" * 70)
        report = classification_report(
            y_test, 
            y_pred, 
            zero_division=0,
            digits=3
        )
        print(report)
        
        if show_per_class:
            # Análisis por clase
            print("\nRendimiento por Clase:")
            print("-" * 70)
            
            report_dict = classification_report(
                y_test, 
                y_pred, 
                output_dict=True,
                zero_division=0
            )
            
            # Ordenar por soporte (número de muestras)
            class_metrics = []
            for class_name in self.label_encoder.classes_:
                if class_name in report_dict:
                    metrics = report_dict[class_name]
                    class_metrics.append({
                        'clase': class_name,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1-score': metrics['f1-score'],
                        'support': metrics['support']
                    })
            
            metrics_df = pd.DataFrame(class_metrics)
            metrics_df = metrics_df.sort_values('support', ascending=False)
            
            print(f"{'Clase':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
            print("-" * 70)
            for _, row in metrics_df.iterrows():
                print(f"{row['clase']:<35} "
                      f"{row['precision']:>6.3f} "
                      f"{row['recall']:>6.3f} "
                      f"{row['f1-score']:>6.3f} "
                      f"{int(row['support']):>8,}")
        
        # Matriz de confusión (simplificada para muchas clases)
        print("\n" + "="*70)
        print("MATRIZ DE CONFUSIÓN")
        print("="*70)
        cm = confusion_matrix(y_test, y_pred)
        classes = self.label_encoder.classes_
        
        # Mostrar solo el número correcto vs incorrecto por clase
        print(f"{'Clase':<35} {'Correctos':>12} {'Incorrectos':>12} {'Total':>12}")
        print("-" * 70)
        
        for i, class_name in enumerate(classes):
            correct = cm[i, i]
            total = cm[i, :].sum()
            incorrect = total - correct
            if total > 0:
                print(f"{class_name:<35} {correct:>12,} {incorrect:>12,} {total:>12,}")
        
        return y_pred
    
    def get_feature_importance(self, top_n=10):
        """
        Obtiene las palabras más importantes para cada categoría
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        classes = self.label_encoder.classes_
        
        print("\n" + "="*70)
        print("PALABRAS MÁS IMPORTANTES POR CATEGORÍA")
        print("="*70)
        
        for i, category in enumerate(classes):
            print(f"\n{category}:")
            print("-" * 50)
            
            if len(classes) == 2:
                coef = self.svm_model.coef_[0]
            else:
                coef = self.svm_model.coef_[i]
            
            top_indices = np.argsort(coef)[-top_n:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            top_scores = [coef[idx] for idx in top_indices]
            
            for word, score in zip(top_words, top_scores):
                print(f"  {word:25s} ({score:7.4f})")
    
    def save_model(self, path='ticket_classifier_svm.pkl'):
        """Guarda el modelo completo"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado.")
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'C': self.C,
            'class_distribution': self.class_distribution
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Modelo guardado exitosamente en: {path}")
    
    def load_model(self, path='ticket_classifier_svm.pkl'):
        """Carga un modelo guardado"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.C = model_data['C']
        self.class_distribution = model_data.get('class_distribution')
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.svm_model = self.pipeline.named_steps['svm']
        self.is_trained = True
        
        print(f"✓ Modelo cargado exitosamente desde: {path}")


# ============================================================================
# SCRIPT DE ENTRENAMIENTO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("CLASIFICADOR DE TICKETS - SISTEMA DE SOPORTE")
    print("="*70)
    print()
    
    # CONFIGURACIÓN
    JSONL_FILE = 'tickets_cleaned.jsonl'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 1. Cargar datos
    print("PASO 1: CARGA DE DATOS")
    print("-" * 70)
    
    df = TicketClassifierSVM.load_jsonl(JSONL_FILE)
    
    # Verificar datos
    print(f"\nColumnas encontradas: {list(df.columns)}")
    print(f"Total de registros: {len(df):,}")
    print(f"Categorías únicas: {df['type'].nunique()}")
    
    # 2. Dividir datos (estratificado para mantener proporciones)
    print("\n" + "="*70)
    print("PASO 2: DIVISIÓN DE DATOS")
    print("-" * 70)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['type']  # Mantiene proporciones en train/test
    )
    
    print(f"Training set: {len(train_df):,} tickets ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Test set: {len(test_df):,} tickets ({TEST_SIZE*100:.0f}%)")
    
    # 3. Inicializar clasificador
    print("\n" + "="*70)
    print("PASO 3: INICIALIZACIÓN DEL MODELO")
    print("-" * 70)
    
    classifier = TicketClassifierSVM(
        max_features=15000,
        ngram_range=(1, 2),
        C=1.0
    )
    
    print("Configuración:")
    print(f"  - Max features: {classifier.max_features:,}")
    print(f"  - N-grams: {classifier.ngram_range}")
    print(f"  - Regularización C: {classifier.C}")
    print(f"  - Balance de clases: Activado")
    
    # 4. Preparar textos
    X_train = classifier.combine_text(train_df)
    y_train = train_df['type']
    
    X_test = classifier.combine_text(test_df)
    y_test = test_df['type']
    
    # 5. Entrenar modelo
    print("\n" + "="*70)
    print("PASO 4: ENTRENAMIENTO")
    print("-" * 70)
    
    classifier.train(X_train, y_train)
    
    # 6. Evaluar modelo
    print("\n" + "="*70)
    print("PASO 5: EVALUACIÓN")
    print("-" * 70)
    
    y_pred = classifier.evaluate(X_test, y_test, show_per_class=True)
    
    # 7. Palabras importantes
    classifier.get_feature_importance(top_n=8)
    
    # 8. Guardar modelo
    print("\n" + "="*70)
    print("PASO 6: GUARDAR MODELO")
    print("-" * 70)
    
    classifier.save_model('ticket_classifier_svm.pkl')
    
    """
    # 9. Ejemplo de predicción
    print("\n" + "="*70)
    print("EJEMPLO DE PREDICCIÓN")
    print("="*70)
    
    nuevos_tickets = pd.DataFrame({
        'name': [
            'Instalar paqueteria',
            'Sistema caído',
            'Cambio urgente en producción',
            'Duda sobre procedimiento'
        ],
        'description': [
            'Necesito instalar Office en mi computadora',
            'El servidor principal no responde, usuarios sin acceso',
            'Requerimos actualizar la base de datos de producción inmediatamente',
            'Cómo puedo solicitar un nuevo equipo?'
        ]
    })
    
    X_nuevos = classifier.combine_text(nuevos_tickets)
    predicciones = classifier.predict_with_confidence(X_nuevos)
    print()
    for i in range(len(nuevos_tickets)):
        print(f"Ticket {i+1}:")
        print(f"  Nombre: {nuevos_tickets.iloc[i]['name']}")
        print(f"  → Predicción: {predicciones.iloc[i]['prediction']}")
        print(f"  → Confianza: {predicciones.iloc[i]['confidence_score']:.3f}")
        print()
    """
    print("="*70)
    print("PROCESO COMPLETADO")
    print("="*70)