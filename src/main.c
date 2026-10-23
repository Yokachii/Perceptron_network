#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {

    Perceptron *p = new_perceptron(3,0.1,4);

    free_perceptron(p);
}