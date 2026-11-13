// gcc knn-iris.c sml.c tk_draw.c -o knn-iris -lTensors_cpu -lm -ltcl -ltk
// ./knn
#include <stdio.h>
#include "../book/sml.h"

NDArrayHandle euclidean_distance(NDArrayHandle train, NDArrayHandle test) {
    NDArrayHandle a = expand_dims(train, 1), b = expand_dims(test, 0);
    NDArrayHandle diff = sub(a, b), squared = power(diff, 2);
    NDArrayHandle total = sum(squared, axis(-1)), distances = sqrt(total);
    free_tensors(a, b, diff, squared, total);
    return distances;
}
NDArrayHandle get_neighbors(NDArrayHandle train, NDArrayHandle test, int k) {
    int features = shape_dim(train, 1) - 1;
    NDArrayHandle train_x = slice_axis(train, 1, 0, features);
    NDArrayHandle test_x = slice_axis(test, 1, 0, features);
    NDArrayHandle distances = euclidean_distance(train_x, test_x);
    NDArrayHandle sorted = argsort(distances, 0), indices = slice_axis(sorted, 0, 0, k);
    NDArrayHandle neighbors = take(train, indices);
    free_tensors(train_x, test_x, distances, sorted, indices);
    return neighbors;
}
NDArrayHandle predict(NDArrayHandle neighbors, int classes) {
    int column = shape_dim(neighbors, 2) - 1;
    NDArrayHandle last = slice_axis(neighbors, 2, column, column + 1);
    NDArrayHandle labels = reshape(last, shape(shape_dim(neighbors, 0), shape_dim(neighbors, 1)));
    NDArrayHandle integers = astype(labels, "int32"), encoded = one_hot(integers, classes);
    NDArrayHandle votes = sum(encoded, axis(0)), winners = argmax(votes, -1);
    NDArrayHandle predicted = astype(winners, "int32");
    free_tensors(last, labels, integers, encoded, votes, winners);
    return predicted;
}
int main(void) {
    const char *filename = "../data/iris.csv";
    CSV_Container csv = load_csv(filename, ",", 1);
    if (!csv.dataset) {
        free_csv_container(&csv);
        return 1;
    }
    int k = 12;
    int n_folds = 10;
    if (n_folds < 2 || n_folds > csv.rows || k < 1 ||
        k > (n_folds - 1) * (csv.rows / n_folds)) {
        fprintf(stderr, "Numero de particiones o vecinos invalido.\n");
        free_csv_container(&csv);
        return 1;
    }
    seed(1);
    NDArrayHandle folds = cross_validation_split(csv.dataset, n_folds);
    NDArrayHandle global_actual = NULL, global_predicted = NULL;
    float total_accuracy = 0.0f;
    printf("Dataset: %s (%d filas, %d atributos)\n", filename, csv.rows, csv.cols - 1);
    printf("Validacion cruzada: %d particiones | k: %d | Semilla: 1\n", n_folds, k);
    for (int i = 0; i < n_folds; i++) {
        NDArrayHandle test = slice(folds, i, ":", ":");
        NDArrayHandle remaining = delete(folds, i, 0);
        NDArrayHandle train = reshape(remaining, shape(-1, csv.cols));
        NDArrayHandle neighbors = get_neighbors(train, test, k);
        NDArrayHandle labels = slice(test, ":", csv.cols - 1);
        NDArrayHandle actual = astype(labels, "int32");
        NDArrayHandle predicted = predict(neighbors, csv.label_count);
        NDArrayHandle classes = NULL, matrix = NULL;
        unpack_tensors(confusion_matrix(actual, predicted), classes, matrix);
        float accuracy = accuracy_metric(actual, predicted);
        total_accuracy += accuracy;
        printf("\nParticion %d | Entrenamiento: %d | Prueba: %d\n", i + 1, len(train), len(test));
        printf("Actual %s\n", asstr(actual));
        printf("Predicted %s\n", asstr(predicted));
        printf("Exactitud: %.2f%%\n", accuracy);
        printf("Matriz de confusion %s\n", asstr(matrix));
        if (i == 0) {
            global_actual = actual;
            global_predicted = predicted;
        } else {
            NDArrayHandle combined_actual = concat(0, global_actual, actual);
            NDArrayHandle combined_predicted = concat(0, global_predicted, predicted);
            free_tensors(global_actual, global_predicted, actual, predicted);
            global_actual = combined_actual;
            global_predicted = combined_predicted;
        }
        free_tensors(test, remaining, train, neighbors, labels, classes, matrix);
    }
    printf("\nExactitud promedio: %.2f%%\n", total_accuracy / n_folds);
    NDArrayHandle global_classes = NULL, global_matrix = NULL;
    unpack_tensors(confusion_matrix(global_actual, global_predicted), global_classes, global_matrix);
    printf("Exactitud global (%d muestras): %.2f%%\n", len(global_actual),
           accuracy_metric(global_actual, global_predicted));
    printf("Clases: %s\n", asstr(csv.labels));
    printf("Matriz de confusion global (filas=real, columnas=prediccion):\n%s\n", asstr(global_matrix));
    free_tensors(folds, global_actual, global_predicted, global_classes, global_matrix);
    free_csv_container(&csv);
    return 0;
}
