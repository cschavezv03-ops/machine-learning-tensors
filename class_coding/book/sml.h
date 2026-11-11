#ifndef SML_H
#define SML_H

#include "Tensors.h"

// Chapter 01
typedef struct {
    NDArrayHandle dataset; // Tensor generado por la librería
    char** header;         // Lista de nombres de columnas
    int header_count;      // Cantidad de columnas en el header
    char** labels;         // Mapeo de categorías: labels[id] -> "nombre_categoria"
    int label_count;       // Cantidad de categorías únicas detectadas
    int rows;              // Número de filas leídas
    int cols;              // Número de columnas leídas
} CSV_Container;

static float str_column_to_int(CSV_Container* res, const char* str, int len);
CSV_Container load_csv(const char* file_path, const char* delimiter, int has_header);
void free_csv_container(CSV_Container* res);

// Chapter 02
NDArrayHandle dataset_minmax(NDArrayHandle dataset);
void normalize_dataset(NDArrayHandle X, NDArrayHandle minmax);
void standardize_dataset(NDArrayHandle X, NDArrayHandle means, NDArrayHandle stdevs);

// Chapter 03
NDArrayList train_test_split(NDArrayHandle dataset, float split_ratio);
NDArrayHandle cross_validation_split(NDArrayHandle dataset, int n_folds);
NDArrayHandle count_labels(NDArrayHandle dataset, int num_classes);
void print_formatted_counts(NDArrayHandle counts, char** labels, int label_count);

// Chapter 04
float accuracy_metric(NDArrayHandle actual, NDArrayHandle predicted);
NDArrayList confusion_matrix(NDArrayHandle actual, NDArrayHandle predicted);
float mae_metric(NDArrayHandle actual, NDArrayHandle predicted);
float rmse_metric(NDArrayHandle actual, NDArrayHandle predicted);
NDArrayList perf_metrics(NDArrayHandle actual, NDArrayHandle predicted_prob, NDArrayHandle threshold, float pos_cls);
float trapz(NDArrayHandle x, NDArrayHandle y);
void plot_roc_curve(NDArrayHandle sorted_fprs, NDArrayHandle sorted_tprs, float auc);

#endif // SML_H