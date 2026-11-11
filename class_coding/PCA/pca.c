//Compile: gcc pca.c ../book/sml.c ../book/tk_draw.c -o pca -lTensors_cpu -lm -ltcl -ltk
//Autor: Sebastian Chavez
#include <stdio.h>
#include "Tensors.h"
#include "../book/sml.h"

void main(void){
    CSV_Container iris = load_csv("../data/iris.csv", ",", 1);
    NDArrayHandle X = slice(iris.dataset, ":", range(0, -1));
    NDArrayHandle y = slice(iris.dataset, ":", -1);
    NDArrayHandle X_first = slice(X, range(0, 5));
    NDArrayHandle X_mean = mean(X, axis(0));
    NDArrayHandle X_std = std(X, axis(0));
    printf("X:\n%s\n", asstr(X_first));
    printf("y:\n%s\n", asstr(y));
    printf("Mean: %s\n", asstr(X_mean));
    printf("Deviation: %s\n", asstr(X_std));
    NDArrayHandle lo = min(X, axis(0));
    NDArrayHandle hi = max(X, axis(0));
    NDArrayHandle minmax = stack(1, lo, hi);
    NDArrayHandle span = sub(hi, lo);
    NDArrayHandle offset = sub(X, lo);
    NDArrayHandle norm = div(offset, span);
    NDArrayHandle norm_first = slice(norm, range(0, 5));
    printf("Matriz Min-Max per column:\n%s\n", asstr(minmax));
    printf("Normalized:\n%s\n", asstr(norm_first));
    NDArrayHandle mu = mean(norm, axis(0));
    NDArrayHandle centered = sub(norm, mu);
    NDArrayHandle centered_first = slice(centered, range(0, 5));
    printf("Mean centered data:\n%s\n", asstr(centered_first));
    double scale = 1.0 / (len(X) - 1);
    NDArrayHandle C = linalg_syrk(centered, 1, scale);
    printf("Covariance matrix:\n%s\n", asstr(C));
    NDArrayHandle vals = NULL;
    NDArrayHandle vecs = NULL;
    unpack_tensors(eig(C), vals, vecs);
    printf("Eigen values:\n%s\n", asstr(vals));
    printf("Eigen vectors:\n%s\n", asstr(vecs));
    NDArrayHandle ascending = argsort(vals, 0);
    NDArrayHandle sort_idx = flip(ascending, axis(0));
    NDArrayHandle values = take(vals, sort_idx);
    NDArrayHandle vecs_T = transpose(vecs);
    NDArrayHandle vectors = take(vecs_T, sort_idx);
    printf("Sorted Eigen values:\n%s\n", asstr(values));
    printf("Sorted Eigen vectors:\n%s\n", asstr(vectors));
    NDArrayHandle total = sum(values, axis(0), NULL, 1);
    NDArrayHandle explained = div(values, total);
    NDArrayHandle cumulative = cumsum(explained);
    printf("Explained variance:\n%s\n", asstr(explained));
    printf("Cumulative variance:\n%s\n", asstr(cumulative));
    NDArrayHandle Z = dot(centered, vectors);
    NDArrayHandle first = slice(Z, range(0, 5));
    printf("Transformed data:\n%s\n", asstr(first));
    free_tensors(X, y, X_first, X_mean, X_std, lo, hi, minmax, span, offset);
    free_tensors(norm, norm_first, mu, centered, centered_first, C);
    free_tensors(vals, vecs, ascending, sort_idx, values, vecs_T, vectors);
    free_tensors(total, explained, cumulative, Z, first);
    free_csv_container(&iris);
}
