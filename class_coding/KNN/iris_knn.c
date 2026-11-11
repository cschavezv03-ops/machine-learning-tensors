#include <stdio.h>
#include "Tensors.h"
#include "../book/sml.h"

NDArrayHandle euclidian_distance(NDArrayHandle train, NDArrayHandle test) {
    NDArrayHandle diff = expand_dims(train, 1);
    NDArrayHandle test_exp = expand_dims(test, 0);
    diff = sub(diff, test_exp);
    diff = square(diff);
    diff = sum(diff, axis(-1));
    return sqrt(diff);
}

NDArrayHandle get_neighbors(NDArrayHandle train, NDArrayHandle test, int k) {
    NDArrayHandle trainX = slice(train, ":", range(0, -1));
    NDArrayHandle testX = slice(test, ":", range(0, -1));
    NDArrayHandle distances = euclidian_distance(trainX, testX);
    NDArrayHandle indices = argsort(distances, 0);
    indices = slice(indices, range(0, k), ":");
    return take(train, indices);
}

NDArrayHandle predict(NDArrayHandle neighbors) {
    NDArrayHandle labels = slice(neighbors, ":", ":", -1);
    labels = astype(labels, "int32");
    labels = one_hot(labels, 3);
    labels = sum(labels, axis(0));
    return argmax(labels, -1);
}

int main(void) {
    CSV_Container iris = load_csv("../data/iris.csv", ",", 1);
    if (!iris.dataset) return 1;

    NDArrayHandle train = NULL, test = NULL;
    seed(1);
    unpack_tensors(train_test_split(iris.dataset, 0.8), train, test);

    NDArrayHandle neighbors = get_neighbors(train, test, 2);
    NDArrayHandle predicted = predict(neighbors);
    NDArrayHandle actual = astype(slice(test, ":", -1), "int32");
    NDArrayList result = confusion_matrix(actual, predicted);
    NDArrayHandle classes = NULL, matrix = NULL;
    unpack_tensors(result, classes, matrix);

    printf("Clases: %s\n", asstr(classes));
    printf("Etiquetas reales: %s\n", asstr(actual));
    printf("Predicciones KNN: %s\n", asstr(predicted));
    printf("Matriz de confusion (filas=real, columnas=prediccion):\n%s\n", asstr(matrix));
    NDArrayHandle suma = sum(matrix);
    printf("Suma de la matriz de confusion: %s\n", asstr(suma));

    free_tensors(train, test, neighbors, predicted, actual, classes, matrix);
    free_csv_container(&iris);
    return 0;
}