#include <stdio.h>
#include "Tensors.h"

NDArrayHandle euclidian_distance(NDArrayHandle train, NDArrayHandle test){
    NDArrayHandle diff = expand_dims(train, 1);
    NDArrayHandle test_exp = expand_dims(test, 0);
    diff = sub(diff, test_exp);
    diff = square(diff);
    diff = sum(diff, axis(-1));
    return sqrt(diff);
}

NDArrayHandle get_neighbors(NDArrayHandle train, NDArrayHandle test, int k){
    NDArrayHandle trainX = slice(train, ":", range(0, -1));
    NDArrayHandle testX = slice(test, ":", range(0, -1));
    NDArrayHandle distances = euclidian_distance(trainX, testX);
    NDArrayHandle indices = argsort(distances, 0);
    indices = slice(indices, range(0, k), ":");
    return take(train, indices);
}

NDArrayHandle predict(NDArrayHandle neighbors){
    NDArrayHandle labels = slice(neighbors, ":", ":", -1);
    labels = astype(labels, "int32");
    labels = one_hot(labels, 2);
    labels = sum(labels, axis(0));
    return argmax(labels, -1);
}

void main(void){
    NDArrayHandle train = array(float32(1.4, 2.3, 0, 7.6, 3.5, 1, 7.6, 2.7, 1, 6.9, 1.7, 1, 1.3, 1.8, 0),shape(5, 3));
    NDArrayHandle test = array(float32(3.0, 3.0, 0, 8.6, -0.2, 1, 2.7, 2.5, 0, 3.3, 4.4, 0, 5.3, 2.0, 1),shape(5, 3));
    int k = 3;
    NDArrayHandle neighbors = get_neighbors(train, test, k);
    printf("Neighbors:\n%s",  asstr(neighbors));
    NDArrayHandle predictions = predict(neighbors);
    printf("Expected:\n%s\n", asstr(slice(test, ":", -1)));
    printf("Got:\n%s\n", asstr(predictions));
    free_tensors(train, test, neighbors, predictions);
}