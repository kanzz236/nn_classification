
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "config.h"

typedef struct {
    double W1[H1][NUM_FEATURES], b1[H1];
    double W2[H2][H1], b2[H2];
    double W3[H2], b3;

    double vW1[H1][NUM_FEATURES], vb1[H1];
    double vW2[H2][H1], vb2[H2];
    double vW3[H2], vb3;
} NeuralNetwork;

double relu(double x);
double relu_derivative(double x);
double sigmoid(double x);

double rand_gaussian(double mean, double stddev);

void initialize_network(NeuralNetwork* nn);

double forward(
    NeuralNetwork* nn,
    const double input[],
    double z1[],
    double a1[],
    double z2[],
    double a2[]
);

#endif
