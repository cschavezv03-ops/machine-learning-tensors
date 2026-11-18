// gcc k-means.c ../../../book/sml.c ../../../book/tk_draw.c -o k-means -lTensors_cpu -lm -ltcl -ltk
// Autor: Sebastian Chavez
#include <stdio.h>
#include <math.h>
#include "Tensors.h"
#include "../../../book/sml.h"

void show_sample(NDArrayHandle X) {
    NDArrayHandle sample = slice(X, range(0, 5));
    printf("%s\n", asstr(sample));
    free_tensors(sample);
}

NDArrayList init_random_centroids(NDArrayHandle X, int K) {
    NDArrayHandle indices = arange(len(X));
    NDArrayHandle shuffled = shuffle(indices);
    NDArrayHandle positions = slice(shuffled, range(0, K));
    NDArrayHandle centroids = take(X, positions);
    free_tensors(indices, shuffled);
    return tolist(centroids, positions);
}

NDArrayHandle assign_cluster(NDArrayHandle X, NDArrayHandle centroids) {
    NDArrayHandle samples = expand_dims(X, 1);
    NDArrayHandle centers = expand_dims(centroids, 0);
    NDArrayHandle difference = sub(samples, centers);
    NDArrayHandle squared = square(difference);
    NDArrayHandle dists = sum(squared, axis(2));
    NDArrayHandle assignments = argmin(dists, 1);
    free_tensors(samples, centers, difference, squared, dists);
    return assignments;
}

NDArrayHandle new_centroids(NDArrayHandle X, NDArrayHandle assignments, int K) {
    NDArrayHandle encoded = one_hot(assignments, K);
    NDArrayHandle encoded_T = transpose(encoded);
    NDArrayHandle counts = sum(encoded_T, axis(1), NULL, 1);
    NDArrayHandle sums = dot(encoded_T, X);
    NDArrayHandle divisor = broadcast_maximum(counts, 1);
    NDArrayHandle averages = div(sums, divisor);
    free_tensors(encoded, counts, encoded_T, sums, divisor);
    return averages;
}

double sse(NDArrayHandle X, NDArrayHandle assignments, NDArrayHandle centroids) {
    NDArrayHandle centroid = take(centroids, assignments);
    NDArrayHandle errors = sub(X, centroid);
    NDArrayHandle squared = square(errors);
    NDArrayHandle total = sum(squared);
    double result = asscalar(total);
    free_tensors(centroid, errors, squared, total);
    return result;
}

NDArrayList fit(NDArrayHandle X, int K, int max_iter, double tol, int verbose) {
    NDArrayHandle assignments = NULL, centroids = NULL, init_centroids_pos = NULL;
    unpack_tensors(init_random_centroids(X, K), centroids, init_centroids_pos);
    NDArrayHandle historical_sse = NULL;
    double prev_sse_val = 0;
    for (int iter = 1; iter <= max_iter; iter++) {
        free_tensors(assignments);
        assignments = assign_cluster(X, centroids);
        free_tensors(centroids);
        centroids = new_centroids(X, assignments, K);
        double current_sse_val = sse(X, assignments, centroids);
        NDArrayHandle error = array(float32(current_sse_val), shape(1));
        if (iter == 1) {
            historical_sse = error;
        } else {
            NDArrayHandle combined = concat(0, historical_sse, error);
            free_tensors(historical_sse, error);
            historical_sse = combined;
        }
        double delta = fabs(prev_sse_val - current_sse_val);
        if (iter > 1 && delta < tol) {
            if (verbose)
                printf("k-Means convergence achieved in iteration %d (Delta: %g)\n", iter, delta);
            break;
        }
        prev_sse_val = current_sse_val;
    }
    return tolist(assignments, centroids, historical_sse, init_centroids_pos);
}

int main(void) {
    NDArrayHandle train, test;
    CSV_Container csv = load_csv("../data/iris.csv", ",", 1);
    if (!csv.dataset) return 1;
    int semilla = 1;
    seed(semilla);
    unpack_tensors(train_test_split(csv.dataset, 0.8), train, test);
    printf("Train:\n");
    show_sample(train);
    printf("Test:\n");
    show_sample(test);
    NDArrayHandle X_train = slice(train, ":", range(0, -1));
    NDArrayHandle y_train = slice(train, ":", -1);
    NDArrayHandle X_test = slice(test, ":", range(0, -1));
    NDArrayHandle y_test = slice(test, ":", -1);
    printf("X_train:\n");
    show_sample(X_train);
    printf("X_test:\n");
    show_sample(X_test);
    NDArrayHandle X = slice(csv.dataset, ":", range(0, -1));
    NDArrayHandle minmax = dataset_minmax(X);
    normalize_dataset(X_train, minmax);
    normalize_dataset(X_test, minmax);
    printf("Matriz Min-Max por columna:\n%s\n", asstr(minmax));
    printf("X_train_norm:\n");
    show_sample(X_train);
    printf("X_test_norm:\n");
    show_sample(X_test);
    int K = 3;
    NDArrayHandle centroids, init_centroids_pos;
    unpack_tensors(init_random_centroids(X_train, K), centroids, init_centroids_pos);
    printf("centroids:\n%s\n", asstr(centroids));
    printf("init_centroids_pos:\n%s\n", asstr(init_centroids_pos));
    NDArrayHandle assignments = assign_cluster(X_train, centroids);
    printf("assignments:\n%s\n", asstr(assignments));
    free_tensors(centroids);
    centroids = new_centroids(X_train, assignments, K);
    printf("new_centroids:\n%s\n", asstr(centroids));
    printf("sse:\n%g\n", sse(X_train, assignments, centroids));
    free_tensors(assignments, centroids, init_centroids_pos);
    NDArrayHandle all_sse;
    unpack_tensors(fit(X_train, K, 20, 1e-4, 1), assignments, centroids, all_sse, init_centroids_pos);
    printf("sse:\n%s\n", asstr(all_sse));
    printf("centroids:\n%s\n", asstr(centroids));
    printf("init_centroids_pos:\n%s\n", asstr(init_centroids_pos));
    NDArrayHandle clusters = assign_cluster(X_train, centroids);
    printf("clusters:%s\n", asstr(clusters));
    free_tensors(clusters);
    printf("y: %s\n", asstr(y_train));
    printf("assignments: %s\n", asstr(assignments));
    NDArrayHandle classes, matrix;
    unpack_tensors(confusion_matrix(y_train, assignments), classes, matrix);
    printf("Confusion matrix:\n%s\n", asstr(matrix));
    free_tensors(classes, matrix);
    NDArrayHandle sort_idx = array(int32(1, 2, 0), shape(3));
    NDArrayHandle sorted_centroids = take(centroids, sort_idx);
    printf("sorted_centroids:\n%s\n", asstr(sorted_centroids));
    clusters = assign_cluster(X_test, sorted_centroids);
    printf("clusters:%s\n", asstr(clusters));
    unpack_tensors(confusion_matrix(y_test, clusters), classes, matrix);
    printf("Confusion matrix:\n%s\n", asstr(matrix));
    free_tensors(train, test, X_train, y_train, X_test, y_test, X, minmax);
    free_tensors(assignments, centroids, all_sse, init_centroids_pos, sort_idx);
    free_tensors(sorted_centroids, clusters, classes, matrix);
    free_csv_container(&csv);
    return 0;
}
