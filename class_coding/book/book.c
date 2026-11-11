#include <stdio.h>
#include <stdlib.h>
#include "tk_draw.h"
#include "sml.h"
#include "Tensors.h"

//  gcc book.c sml.c tk_draw.c -o book -lTensors_cpu -lm -ltcl -ltk
// ./book

// Declaraciones previas de las funciones de cada capítulo
void Chapter01_Load_Data_From_CSV();
void Chapter02_Scale_Machine_Learning_Data();
void Chapter03_Algorithm_Evaluation_Methods();
void Chapter04_Evaluation_Metrics();

int main(int argc, char *argv[]) {
    // Si no se proporciona un argumento, se muestra la ayuda
    if (argc < 2) {
        printf("Uso: %s <numero_de_capitulo>\n\n", argv[0]);
        printf("Capítulos disponibles:\n");
        printf("  1: Load Data From CSV\n");
        printf("  2: Scale Machine Learning Data\n");
        printf("  3: Algorithm Evaluation Methods\n");
        printf("  4: Evaluation Metrics\n");
        return 1;
    }

    // Convertir el argumento de texto a entero
    int option = atoi(argv[1]);

    switch (option) {
        case 1:
            Chapter01_Load_Data_From_CSV();
            break;
        case 2:
            Chapter02_Scale_Machine_Learning_Data();
            break;
        case 3:
            Chapter03_Algorithm_Evaluation_Methods();
            break;
        case 4:
            Chapter04_Evaluation_Metrics();
            break;
        default:
            printf("Error: Opción inválida (%d). Seleccione un capítulo válido.\n", option);
            return 1;
    }

    return 0;
}

void Chapter01_Load_Data_From_CSV() {
  // 1. Cargar datos del archivo CSV
  CSV_Container csv_data = load_csv("../data/iris.csv", ",", 1);

  // 2. Imprimir información básica (# filas, # columnas, encabezados de columnas, y clases)
  printf("Filas leídas: %d\n", csv_data.rows);
  printf("Columnas leídas: %d\n", csv_data.cols);
  printf("%d columnas:%s\n", csv_data.header_count, asstr(csv_data.header));
  printf("%d clases:%s\n", csv_data.label_count, asstr(csv_data.labels));

  // 4. Extraer porción del tensor con `slice` e imprimirla con `asstr`
  if (csv_data.dataset) {
    NDArrayList portions = declare(3);
  
    // Extraer tres porciones de 5 filas correspondientes a cada clase
    idx(portions, 0) = slice(csv_data.dataset, range(0, 5), ":");
    idx(portions, 1) = slice(csv_data.dataset, range(50, 55), ":");
    idx(portions, 2) = slice(csv_data.dataset, range(100, 105), ":");

    // Imprimir porción mediante asstr
    printf("\n--- Muestra del Tensor (primeras 5 filas) ---\n%s\n", asstr(idx(portions, 0)));
    printf("\n--- Muestra del Tensor (5 filas a partir del registro 50) ---\n%s\n", asstr(idx(portions, 1)));
    printf("\n--- Muestra del Tensor (5 filas a partir del registro 100) ---\n%s\n", asstr(idx(portions, 2)));

    // Liberar el tensor temporal creado por la operación slice
    free_tensors(portions);
  }

  // 5. Liberar memoria del dataset y header
  free_csv_container(&csv_data);
}

void Chapter02_Scale_Machine_Learning_Data() {
  // === 1. Prueba de MinMax y Normalización ===
  NDArrayHandle dataset, minmax, means, stdevs, X, y = NULL;
  dataset = array(float32(50, 30, 20, 90), shape(2, 2));
  printf("Dataset: %s\n", asstr(dataset));

  minmax = dataset_minmax(dataset);
  printf("Minimax: %s\n", asstr(minmax));

  normalize_dataset(dataset, minmax);
  printf("Normalized: %s\n\n", asstr(dataset));

  free_tensors(dataset, minmax);

  // === 2. Carga y normalización del dataset diabetes (Pima) ===
  const char* filename = "../data/pima-indians-diabetes.csv";
  CSV_Container csv = load_csv(filename, ",", 1);

  X = slice(csv.dataset, ":", range(0, csv.cols - 1));
  y = slice(csv.dataset, ":", csv.cols - 1);

  printf("Loaded data file %s with %d rows and %d columns.\n\n", filename, len(csv.dataset), csv.cols);
  printf("X: %s\n", asstr(X));

  minmax = dataset_minmax(X);
  normalize_dataset(X, minmax);
  printf("Normalized: %s\n\n", asstr(X));

  free_tensors(X, y, minmax);
  free_csv_container(&csv); // Libera csv.dataset, header, etc.

  // === 3. Medias, desviación estándar y estandarización ===
  dataset = array(float32(50, 30, 20, 90, 30, 50), shape(3, 2));
  printf("Dataset: %s\n", asstr(dataset));

  means  = mean(dataset, axis(0)); // Calcula la media por columna
  stdevs = std(dataset, axis(0), "float32", 0, 1);
  printf("Means: %s\n", asstr(means));
  printf("Stdevs: %s\n", asstr(stdevs));
  
  standardize_dataset(dataset, means, stdevs);
  printf("Standardized: %s\n\n", asstr(dataset));

  free_tensors(dataset, means, stdevs);

  // === 4. Estandarización completa del dataset diabetes ===
  CSV_Container csv2 = load_csv(filename, ",", 1);
  X = slice(csv2.dataset, ":", range(0, csv2.cols - 1));

  minmax = dataset_minmax(X);
  normalize_dataset(X, minmax);

  means  = mean(X, axis(0)); // Calcula la media por columna
  stdevs = std(X, axis(0), "float32", 0, 1);
  printf("Stdevs: %s\n", asstr(stdevs));

  standardize_dataset(X, means, stdevs);
  printf("Standardized diabetes: %s\n", asstr(X));

  free_tensors(X, minmax, means, stdevs);
  free_csv_container(&csv2);
}

void Chapter03_Algorithm_Evaluation_Methods() {
  printf("================================================================================\n");
  printf(" CAPÍTULO 3: MÉTODOS DE REMUESTREO\n");
  printf("================================================================================\n\n");

  // --------------------------------------------------------------------------
  // 1. Demostración train / test split con dataset dummy
  // --------------------------------------------------------------------------
  printf("--- 1. Pruebas de train_test_split (dataset dummy) ---\n");
  
  NDArrayHandle dataset = NULL, train = NULL, test = NULL, folds = NULL;
  
  seed(1);
  
  dataset = arange(1, 11);
  unpack_tensors(train_test_split(dataset, 0.6), train, test);

  printf("Train:\n%s\nTest:\n%s\n", asstr(train), asstr(test));
  free_tensors(train, test);

  // --------------------------------------------------------------------------
  // 2. Demostración cross-validation split con dataset dummy
  // --------------------------------------------------------------------------
  printf("\n--- 2. Pruebas de cross_validation_split (dataset dummy) ---\n");
  
  seed(1);
  
  dataset = reshape(dataset, shape(10, 1));
  folds   = cross_validation_split(dataset, 5);
  printf("Folds (3D tensor):\n%s\n", asstr(folds));
  free_tensors(dataset, folds);

  // --------------------------------------------------------------------------
  // 3. Carga y Procesamiento del dataset real (iris.csv)
  // --------------------------------------------------------------------------
  printf("\n--- 3. Carga de dataset iris.csv ---\n");
  CSV_Container iris_csv = load_csv("../data/iris.csv", ",", 1);
  printf("rows: %d\ncols: %d\n", iris_csv.rows, iris_csv.cols);

  NDArrayHandle iris_dataset = iris_csv.dataset;

  NDArrayHandle iris_sample = slice(iris_dataset, range(0, 5));
  printf("Primeras 5 filas de iris:\n%s\n", asstr(iris_sample));
  free_tensors(iris_sample);

  // --------------------------------------------------------------------------
  // 4. Conteo Vectorizado sobre dataset completo
  // --------------------------------------------------------------------------
  printf("\n--- 4. Conteo de clases en el dataset completo ---\n");
  NDArrayHandle iris_counts = count_labels(iris_dataset, iris_csv.label_count);
  printf("Tensor de conteos:\n%s\n", asstr(iris_counts));

  printf("Conteo formateado:\n");
  print_formatted_counts(iris_counts, iris_csv.labels, iris_csv.label_count);
  free_tensors(iris_counts);

  // --------------------------------------------------------------------------
  // 5. Train / test split (80/20) sobre iris dataset y conteos
  // --------------------------------------------------------------------------
  printf("\n--- 5. Particiones train/test (80/20) sobre iris dataset ---\n");
  
  seed(1);
  
  unpack_tensors(train_test_split(iris_dataset, 0.8), train, test);

  printf("train size: %d, test size: %d\n", len(train), len(test));

  NDArrayHandle train_counts = count_labels(train, iris_csv.label_count);
  NDArrayHandle test_counts  = count_labels(test,  iris_csv.label_count);

  printf("Conteos train: ");
  print_formatted_counts(train_counts, iris_csv.labels, iris_csv.label_count);

  printf("Conteos test:  ");
  print_formatted_counts(test_counts, iris_csv.labels,  iris_csv.label_count);

  NDArrayHandle train_sample = slice(train, range(0, 5));
  NDArrayHandle test_sample  = slice(test, range(0, 5));
  printf("\nMuestra Train (0-5):\n%s\n", asstr(train_sample));
  printf("Muestra Test (0-5):\n%s\n", asstr(test_sample));

  free_tensors(train_counts, test_counts, train_sample, test_sample, train, test);

  // --------------------------------------------------------------------------
  // 6. Cross-validation (10 folds) y conteo paralelo (tensor 3D)
  // --------------------------------------------------------------------------
  printf("\n--- 6. Cross-Validation (10 folds) y conteo simultáneo en 3D ---\n");
  
  seed(1);
  
  folds = cross_validation_split(iris_dataset, 10);

  NDArrayHandle folds_sample = slice(folds, range(0, 2));
  printf("Slice de los primeros 2 folds:\n%s\n", asstr(folds_sample));
  free_tensors(folds_sample);

  printf("rows per fold: %d\n\n", shape_dim(folds, 1));

  // Conteo vectorizado en una sola operación matricial sobre los 10 FOLDS en paralelo (eje 1)
  NDArrayHandle fold_counts = count_labels(folds, iris_csv.label_count);
  printf("Distribución de clases de las 10 carpetas al mismo tiempo:\n");
  print_formatted_counts(fold_counts, iris_csv.labels, iris_csv.label_count);

  // --------------------------------------------------------------------------
  // Limpieza total de recursos
  // --------------------------------------------------------------------------
  free_tensors(fold_counts, folds);
  free_csv_container(&iris_csv);
}

void Chapter04_Evaluation_Metrics() {
    // -------------------------------------------------------------------------
    // Test 1: Accuracy Metric
    // -------------------------------------------------------------------------
    
    NDArrayHandle actual = NULL, predicted = NULL, predicted_prob = NULL, stacked = NULL;
    
    actual    = array(int32(0, 0, 0, 0, 0, 1, 1, 1, 1, 1));
    predicted = array(int32(0, 1, 0, 0, 0, 1, 0, 1, 1, 1));

    stacked = stack(1, actual, predicted);
    printf("actual\tpredicted\n%s\n", asstr(stacked));

    float accuracy = accuracy_metric(actual, predicted);
    printf("Accuracy: %0.2f%%\n\n", accuracy);

    free_tensors(actual, predicted, stacked);

    // -------------------------------------------------------------------------
    // Test 2: Confusion Matrix
    // -------------------------------------------------------------------------
    actual    = array(int32(0, 0, 0, 0, 0, 1, 1, 1, 1, 1));
    predicted = array(int32(0, 1, 1, 0, 0, 1, 0, 1, 1, 1));

    NDArrayList cm_res = confusion_matrix(actual, predicted);
    NDArrayHandle clases = NULL, matrix = NULL;
    unpack_tensors(cm_res, clases, matrix);

    printf("clases: %s\n", asstr(clases));
    printf("confusion matrix:\n%s\n\n", asstr(matrix));

    free_tensors(actual, predicted, clases, matrix);

    // -------------------------------------------------------------------------
    // Test 3: MAE & RMSE
    // -------------------------------------------------------------------------
    actual    = array(float32(0.1, 0.2, 0.3, 0.4, 0.5));
    predicted = array(float32(0.11, 0.19, 0.29, 0.41, 0.5));

    stacked = stack(1, actual, predicted);
    printf("actual\tpredicted\n%s\n", asstr(stacked));
    
    printf("mae: %0.3f\n", mae_metric(actual, predicted));
    printf("rmse: %0.3f\n\n", rmse_metric(actual, predicted));

    free_tensors(actual, predicted, stacked);

    // -------------------------------------------------------------------------
    // Test 4: ROC Curve (Perf Metrics) & Trapz (AUC)
    // -------------------------------------------------------------------------
    
    // Cargar datos del archivo CSV
    CSV_Container csv_data = load_csv("../data/model.csv", ",", 1);
    NDArrayHandle actual_probs_cols = slice(csv_data.dataset, ":", range(1, None));
    
    // Obtención de actual y probabilidades
    unpack_tensors(split(actual_probs_cols, 2, 1, 1), actual, predicted_prob);
    free_tensors(actual_probs_cols);

    predicted = gt(predicted_prob, 0.5);
    NDArrayList samples = declare(5);
    idx(samples, 0) = slice(actual, range(0, 5));
    idx(samples, 1) = slice(predicted_prob, range(0, 5));
    idx(samples, 2) = slice(predicted, range(0, 5));
    
    printf("actual=%s\npredicted_prob=%s\npredicted=%s\n", asstr(idx(samples, 0)), asstr(idx(samples, 1)), asstr(idx(samples, 2)));
    
    NDArrayHandle thresholds = linspace(0.0, 1.0, 100);
    NDArrayHandle fprs = NULL, tprs = NULL;
    
    unpack_tensors(perf_metrics(actual, predicted_prob, thresholds, 1.0), fprs, tprs);

    // Ordenar FPR para calcular AUC con trapz
    NDArrayHandle sorted_indices = argsort(fprs, 0, 1); // Ascendente en eje 0
    NDArrayHandle sorted_fprs    = take(fprs, sorted_indices);
    NDArrayHandle sorted_tprs    = take(tprs, sorted_indices);

    float auc = trapz(sorted_fprs, sorted_tprs);
    printf("Area under the ROC curve (AUC): %0.2f\n", auc);
    
    // --- Impresión de valores para verificación ---
    idx(samples, 3) = slice(sorted_fprs, range(20, 30));
    idx(samples, 4) = slice(sorted_tprs, range(20, 30));
    printf("sorted fprs (20 ... 30):\n%s\n", asstr(idx(samples, 3)));
    printf("sorted tprs (20 ... 30):\n%s\n", asstr(idx(samples, 4)));
    printf("Area under the ROC curve (AUC): %0.2f\n\n", auc);
    
    // Renderizado Tk de la curva ROC
    plot_roc_curve(sorted_fprs, sorted_tprs, auc);

    // Liberación de memoria de tensores
    free_tensors(thresholds, actual, predicted_prob, predicted, samples, fprs, tprs, sorted_indices, sorted_fprs, sorted_tprs);
    free_csv_container(&csv_data);
}
