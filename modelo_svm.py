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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    
    def plot_results(self, X_test, y_test, y_pred=None, output_dir='graficas_svm'):
        """
        Crea visualizaciones individuales de los resultados del modelo
        Cada gráfica se guarda en un archivo separado
        
        Args:
            X_test: Features de prueba
            y_test: Etiquetas reales
            y_pred: Predicciones (si None, se calculan)
            output_dir: Directorio donde guardar las imágenes
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido cargado.")
        
        # Crear directorio de salida
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ Directorio creado: {output_dir}")
        
        if y_pred is None:
            print("Generando predicciones...")
            y_pred = self.predict(X_test)
        
        print("\n" + "="*70)
        print("GENERANDO GRÁFICAS INDIVIDUALES")
        print("="*70)
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Calcular métricas
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # =====================================================================
        # GRÁFICA 1: Distribución de clases (Training Set)
        # =====================================================================
        print("1/7 Generando: Distribución de Clases (Training)...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        if self.class_distribution is not None:
            dist_df = self.class_distribution.head(15)
            colors = sns.color_palette("husl", len(dist_df))
            bars = ax.barh(range(len(dist_df)), dist_df.values, color=colors)
            ax.set_yticks(range(len(dist_df)))
            ax.set_yticklabels([str(x)[:30] for x in dist_df.index], fontsize=10)
            ax.set_xlabel('Cantidad de muestras', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Categorías - Training Set (Top 15)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.invert_yaxis()
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(dist_df.values)*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{int(width):,}', ha='left', va='center', fontsize=9)
            
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_distribucion_clases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 2: F1-Score por clase
        # =====================================================================
        print("2/7 Generando: F1-Score por Categoría...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        class_f1 = []
        class_names = []
        for class_name in self.label_encoder.classes_:
            if class_name in report_dict:
                class_f1.append(report_dict[class_name]['f1-score'])
                class_names.append(str(class_name)[:30])
        
        # Ordenar por F1-score
        sorted_indices = np.argsort(class_f1)[::-1][:15]
        top_f1 = [class_f1[i] for i in sorted_indices]
        top_names = [class_names[i] for i in sorted_indices]
        
        # Colores basados en el valor
        colors = [plt.cm.RdYlGn(score) for score in top_f1]
        bars = ax.barh(range(len(top_f1)), top_f1, color=colors)
        ax.set_yticks(range(len(top_f1)))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_title('Top 15 Categorías por F1-Score', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='F1 = 0.5')
        ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='F1 = 0.7')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_f1_score_categorias.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 3: Accuracy Global
        # =====================================================================
        print("3/7 Generando: Accuracy Global...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sizes = [accuracy, 1-accuracy]
        labels = [f'Correctas\n{accuracy:.2%}', f'Incorrectas\n{(1-accuracy):.2%}']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='',
                                           startangle=90, colors=colors, explode=explode,
                                           textprops={'fontsize': 14, 'fontweight': 'bold'},
                                           shadow=True)
        
        ax.set_title(f'Accuracy Global del Modelo: {accuracy:.4f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Agregar leyenda con estadísticas
        total_samples = len(y_test)
        correct_samples = int(accuracy * total_samples)
        incorrect_samples = total_samples - correct_samples
        
        legend_text = f'Total de muestras: {total_samples:,}\n'
        legend_text += f'Correctas: {correct_samples:,}\n'
        legend_text += f'Incorrectas: {incorrect_samples:,}'
        
        ax.text(0.5, -1.3, legend_text, transform=ax.transAxes,
               fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_accuracy_global.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 4: Matriz de Confusión
        # =====================================================================
        print("4/7 Generando: Matriz de Confusión...")
        
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Limitar a las 15 clases más frecuentes
        n_classes = len(self.label_encoder.classes_)
        if n_classes > 15 and self.class_distribution is not None:
            top_classes_idx = self.class_distribution.head(15).index
            class_mask = [c in top_classes_idx for c in self.label_encoder.classes_]
            cm_normalized_plot = cm_normalized[class_mask][:, class_mask]
            labels_plot = [str(c)[:25] for c in self.label_encoder.classes_ if c in top_classes_idx]
            title_suffix = ' (Top 15 Categorías)'
            figsize = (14, 12)
        else:
            cm_normalized_plot = cm_normalized
            labels_plot = [str(c)[:25] for c in self.label_encoder.classes_]
            title_suffix = ''
            figsize = (max(12, n_classes * 0.8), max(10, n_classes * 0.7))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm_normalized_plot, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=labels_plot, yticklabels=labels_plot,
                   ax=ax, cbar_kws={'label': 'Proporción de Predicciones'})
        
        ax.set_xlabel('Categoría Predicha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Categoría Real', fontsize=12, fontweight='bold')
        ax.set_title(f'Matriz de Confusión Normalizada{title_suffix}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_matriz_confusion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 5: Comparación Precision, Recall, F1
        # =====================================================================
        print("5/7 Generando: Comparación de Métricas...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metrics_data = []
        for class_name in self.label_encoder.classes_:
            if class_name in report_dict:
                metrics_data.append({
                    'Clase': str(class_name)[:20],
                    'Precision': report_dict[class_name]['precision'],
                    'Recall': report_dict[class_name]['recall'],
                    'F1-Score': report_dict[class_name]['f1-score']
                })
        
        # Ordenar por F1-Score y tomar top 15
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('F1-Score', ascending=False).head(15)
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        bars1 = ax.bar(x - width, metrics_df['Precision'], width, 
                      label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, metrics_df['Recall'], width, 
                      label='Recall', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, 
                      label='F1-Score', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Categorías', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Métricas por Categoría (Top 15)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Clase'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_comparacion_metricas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 6: Support por Categoría
        # =====================================================================
        print("6/7 Generando: Support por Categoría...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        support_data = []
        for class_name in self.label_encoder.classes_:
            if class_name in report_dict:
                support_data.append({
                    'Clase': str(class_name)[:25],
                    'Support': report_dict[class_name]['support']
                })
        
        support_df = pd.DataFrame(support_data).sort_values('Support', ascending=False).head(15)
        
        colors = sns.color_palette("viridis", len(support_df))
        bars = ax.bar(range(len(support_df)), support_df['Support'], color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(support_df)))
        ax.set_xticklabels(support_df['Clase'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Cantidad de Muestras', fontsize=12, fontweight='bold')
        ax.set_xlabel('Categorías', fontsize=12, fontweight='bold')
        ax.set_title('Support por Categoría (Top 15)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_support_categorias.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # GRÁFICA 7: Resumen de Estadísticas
        # =====================================================================
        print("7/7 Generando: Resumen de Estadísticas...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Título
        title_text = "RESUMEN DE ESTADÍSTICAS DEL MODELO"
        ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
               fontsize=18, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Estadísticas generales
        stats_text = f"""
            MÉTRICAS GLOBALES
            {'='*50}

            Total de Muestras de Prueba: {len(y_test):,}
            Número de Clases: {len(self.label_encoder.classes_)}
            Accuracy Global: {accuracy:.4f} ({accuracy*100:.2f}%)


            PROMEDIOS MACRO (sin ponderar por cantidad)
            {'='*50}

            Precision Macro:  {report_dict['macro avg']['precision']:.4f}
            Recall Macro:     {report_dict['macro avg']['recall']:.4f}
            F1-Score Macro:   {report_dict['macro avg']['f1-score']:.4f}


            PROMEDIOS WEIGHTED (ponderados por cantidad)
            {'='*50}

            Precision Weighted:  {report_dict['weighted avg']['precision']:.4f}
            Recall Weighted:     {report_dict['weighted avg']['recall']:.4f}
            F1-Score Weighted:   {report_dict['weighted avg']['f1-score']:.4f}


            ANÁLISIS DE RENDIMIENTO
            {'='*50}

            Clases con F1 > 0.8: {sum(1 for c in self.label_encoder.classes_ if c in report_dict and report_dict[c]['f1-score'] > 0.8)}
            Clases con F1 > 0.6: {sum(1 for c in self.label_encoder.classes_ if c in report_dict and report_dict[c]['f1-score'] > 0.6)}
            Clases con F1 < 0.5: {sum(1 for c in self.label_encoder.classes_ if c in report_dict and report_dict[c]['f1-score'] < 0.5)}
        """
        
        ax.text(0.1, 0.85, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_resumen_estadisticas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*70)
        print("✓ GRÁFICAS GENERADAS EXITOSAMENTE")
        print("="*70)
        print(f"Ubicación: {output_dir}/")
        print(f"Total de gráficas: 7")
        print("\nArchivos generados:")
        print("  1. 01_distribucion_clases.png")
        print("  2. 02_f1_score_categorias.png")
        print("  3. 03_accuracy_global.png")
        print("  4. 04_matriz_confusion.png")
        print("  5. 05_comparacion_metricas.png")
        print("  6. 06_support_categorias.png")
        print("  7. 07_resumen_estadisticas.png")
        print("="*70)
    
    def get_feature_importance(self, top_n=10):
        """Obtiene las palabras más importantes para cada categoría"""
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
    
    # 8. GENERAR GRÁFICAS
    print("\n" + "="*70)
    print("PASO 6: GENERACIÓN DE VISUALIZACIONES")
    print("-" * 70)
    
    classifier.plot_results(X_test, y_test, y_pred, output_dir='graficas_svm')

    # 9. Guardar modelo
    print("\n" + "="*70)
    print("PASO 6: GUARDAR MODELO")
    print("-" * 70)
    
    classifier.save_model('ticket_classifier_svm.pkl')
    
    print("="*70)
    print("PROCESO COMPLETADO")
    print("="*70)