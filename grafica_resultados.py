import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

class TicketClassifierVisualizer:
    """
    Clase para cargar modelo SVM y generar visualizaciones individuales
    """
    
    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self.vectorizer = None
        self.svm_model = None
        self.class_distribution = None
        self.is_trained = False
        
    def load_model(self, path='ticket_classifier_svm.pkl'):
        """Carga un modelo guardado"""
        print("="*70)
        print("CARGANDO MODELO")
        print("="*70)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.class_distribution = model_data.get('class_distribution')
        self.vectorizer = self.pipeline.named_steps['tfidf']
        self.svm_model = self.pipeline.named_steps['svm']
        self.is_trained = True
        
        print(f"✓ Modelo cargado exitosamente desde: {path}")
        print(f"✓ Clases entrenadas: {len(self.label_encoder.classes_)}")
        print(f"✓ Vocabulario TF-IDF: {len(self.vectorizer.vocabulary_):,} términos")
        print()
        
    @staticmethod
    def load_jsonl(file_path):
        """Carga datos desde archivo JSONL"""
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
        print(f"✓ Cargados {len(df):,} registros\n")
        return df
    
    def combine_text(self, df):
        """Combina 'name' y 'description'"""
        name = df['name'].fillna('').astype(str)
        description = df['description'].fillna('').astype(str)
        return name + ' ' + description
    
    def predict(self, X_test):
        """Predice las categorías de nuevos tickets"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido cargado.")
        
        y_pred_encoded = self.pipeline.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
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
            colors = palette_11 = [
                "#080A30",  
                "#7B17FF",  
                "#1F3C88",
                "#00B4D8",
                "#2EC4B6", 
                "#90DBF4",  
                "#FFD166",  
                "#F8961E",  
                "#EF476F",  
                "#CDB4DB",  
                "#ADB5BD"   
            ]
            bars = ax.barh(range(len(dist_df)), dist_df.values, color=colors)
            ax.set_yticks(range(len(dist_df)))
            ax.set_yticklabels([str(x)[:30] for x in dist_df.index], fontsize=10)
            ax.set_xlabel('Number of samples', fontsize=12, fontweight='bold')
            ax.set_title('Category Distribution - Training Set (Top 15)', fontsize=14, fontweight='bold', pad=20)
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
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_palette",
            ["#90DBF4", "#7B17FF"]
        )
        colors = [custom_cmap(score) for score in top_f1]
        bars = ax.barh(range(len(top_f1)), top_f1, color=colors)
        ax.set_yticks(range(len(top_f1)))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_title('Top 15 Categories for F1-Score', fontsize=14, fontweight='bold', pad=20)
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
        labels = [f'Correct classifications\n{accuracy:.2%}', f'Misclassified\n{(1-accuracy):.2%}']
        colors = ['#7B17FF', '#1F3C88']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='',
                                           startangle=90, colors=colors, explode=explode,
                                           textprops={'fontsize': 14, 'fontweight': 'bold'},
                                           shadow=True)
        
        ax.set_title(f'Model Global Accuracy: {accuracy:.4f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Agregar leyenda con estadísticas
        total_samples = len(y_test)
        correct_samples = int(accuracy * total_samples)
        incorrect_samples = total_samples - correct_samples
        
        legend_text = f'Total of samples: {total_samples:,}\n'
        legend_text += f'Correct classifications: {correct_samples:,}\n'
        legend_text += f'Misclassified: {incorrect_samples:,}'
        
        ax.text(0.5, -0.4, legend_text, transform=ax.transAxes,
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
            title_suffix = ' (Top 15 Categories)'
            figsize = (14, 12)
        else:
            cm_normalized_plot = cm_normalized
            labels_plot = [str(c)[:25] for c in self.label_encoder.classes_]
            title_suffix = ''
            figsize = (max(12, n_classes * 0.8), max(10, n_classes * 0.7))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        purple_cmap = LinearSegmentedColormap.from_list(
            "purple_scale",
            ["#F2E9FF", "#B88CFF", "#7B17FF"]
        )

        sns.heatmap(cm_normalized_plot, annot=False, fmt='.2f', cmap=purple_cmap,
                   xticklabels=labels_plot, yticklabels=labels_plot,
                   ax=ax, cbar_kws={'label': 'Proportion of Predictions'},
                   linewidths=0.6, linecolor='#ADB5BD')
        
        ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Real Category', fontsize=12, fontweight='bold')
        ax.set_title(f'Normalized Confusion Matrix{title_suffix}', 
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
                      label='Precision', color='#7B17FF', alpha=0.8)
        bars2 = ax.bar(x, metrics_df['Recall'], width, 
                      label='Recall', color='#1F3C88', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, 
                      label='F1-Score', color='#00B4D8', alpha=0.8)
        
        ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparison of Metrics by Category (Top 15)', 
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


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("VISUALIZADOR DE RESULTADOS - CLASIFICADOR DE TICKETS")
    print("="*70)
    print()
    
    # CONFIGURACIÓN
    MODEL_PATH = 'ticket_classifier_svm.pkl'
    JSONL_FILE = 'tickets_cleaned.jsonl'
    OUTPUT_DIR = 'graficas_svm'
    
    # 1. Inicializar visualizador
    visualizer = TicketClassifierVisualizer()
    
    # 2. Cargar modelo entrenado
    print("PASO 1: CARGAR MODELO")
    print("-" * 70)
    visualizer.load_model(MODEL_PATH)
    
    # 3. Cargar datos de prueba
    print("\nPASO 2: CARGAR DATOS DE PRUEBA")
    print("-" * 70)
    df = visualizer.load_jsonl(JSONL_FILE)
    
    # Usar una porción para test (20% como en el entrenamiento original)
    from sklearn.model_selection import train_test_split
    
    _, test_df = train_test_split(
        df, 
        test_size=0.2,
        random_state=42,
        stratify=df['type']
    )
    
    print(f"Muestras de prueba: {len(test_df):,}")
    print(f"Categorías únicas: {test_df['type'].nunique()}")
    
    # 4. Preparar datos
    print("\nPASO 3: PREPARAR DATOS")
    print("-" * 70)
    X_test = visualizer.combine_text(test_df)
    y_test = test_df['type']
    print(f"✓ Textos combinados: {len(X_test):,}")
    
    # 5. Generar predicciones
    print("\nPASO 4: GENERAR PREDICCIONES")
    print("-" * 70)
    y_pred = visualizer.predict(X_test)
    print(f"✓ Predicciones generadas: {len(y_pred):,}")
    
    # Mostrar accuracy rápido
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 6. Generar todas las gráficas
    print("\nPASO 5: GENERAR VISUALIZACIONES")
    print("-" * 70)
    visualizer.plot_results(X_test, y_test, y_pred, output_dir=OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n💡 Revisa las gráficas en el directorio: {OUTPUT_DIR}/")
    print("="*70)