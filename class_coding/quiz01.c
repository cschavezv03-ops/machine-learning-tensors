#include <stdio.h>
#include "Tensors.h"
int main(void) {
    NDArrayHandle dataset = array(float32(1, 2, 2, 4, 3, 5, 4, 4, 5, 5), shape(5, 2));
    NDArrayHandle suma = sum(dataset, axis(0), "float32", 1);
    NDArrayHandle medias = div(suma, float32(shape_dim(dataset, 0)));
    printf("\nMedia:%s\n", asstr(medias));
    NDArrayHandle desviaciones = sub(dataset, medias);
    NDArrayHandle cuadrado = square(desviaciones);
    NDArrayHandle suma_cuadrados = sum(cuadrado, axis(0), "float32", 1);
    NDArrayHandle varianzas = div(suma_cuadrados, float32(shape_dim(dataset, 0)));
    printf("\nVarianza poblacional:%s\n", asstr(varianzas));
    NDArrayHandle std = sqrt(varianzas);
    printf("\nDesvio estandar muestral:%s\n", asstr(std));
    varianzas = div(suma_cuadrados, float32(shape_dim(dataset, 0) - 1));
    printf("\nVarianza muestral:%s\n", asstr(varianzas));
    std = sqrt(varianzas);
    printf("\nDesvio estandar muestral:%s\n", asstr(std));
    free_tensors(suma, medias, desviaciones, cuadrado, suma_cuadrados, varianzas, std);
    return 0;
}