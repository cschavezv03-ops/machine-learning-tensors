#include <stdio.h>
#include "Tensors.h"

int main(void) {

    NDArrayHandle x = array(
        float32(1, 4, 9, 16),
        shape(2, 2)
    );

    //Algebraic operations
    NDArrayHandle a = sqrt(x);          //Square root
    NDArrayHandle b = square(x);        //Square
    NDArrayHandle c = reciprocal(x);    //Multiplicative inverse
    NDArrayHandle d = cbrt(x);          //Cubic root

    //Logarithmic operations
    NDArrayHandle e = log(x);           //ln
    NDArrayHandle f = log2(x);
    NDArrayHandle g = log10(x);

    //Exponential transformation
    NDArrayHandle h = exp(x);

    printf("ORIGINAL:\n%s\n", asstr(x));

    printf("\nSQRT:\n%s\n", asstr(a));
    printf("\nSQUARE:\n%s\n", asstr(b));
    printf("\nRECIPROCAL:\n%s\n", asstr(c));
    printf("\nCBRT:\n%s\n", asstr(d));

    printf("\nLOG:\n%s\n", asstr(e));
    printf("\nLOG2:\n%s\n", asstr(f));
    printf("\nLOG10:\n%s\n", asstr(g));
    printf("\nEXP:\n%s\n", asstr(h));

    printf("\nEXP SHAPE: %s\n", shape_str(h));

    free_tensors(x, a, b, c, d, e, f, g, h);

    return 0;
}
