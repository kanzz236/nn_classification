
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_network.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
    if (x < -50) return 0;
    if (x > 50) return 1;

    return 1.0 / (1.0 + exp(-x));
}

double rand_gaussian(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    if (u1 < 1e-10) u1 = 1e-10;

    double z0 = sqrt(-2.0 * log(u1)) *
                cos(2.0 * M_PI * u2);

    return mean + z0 * stddev;
}

void initialize_network(NeuralNetwork* nn) {
    memset(nn, 0, sizeof(NeuralNetwork));

    double std_w1 = sqrt(2.0 / NUM_FEATURES);

    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            nn->W1[i][j] = rand_gaussian(0, std_w1);
        }
    }

    double std_w2 = sqrt(2.0 / H1);

    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H1; j++) {
            nn->W2[i][j] = rand_gaussian(0, std_w2);
        }
    }

    double std_w3 = sqrt(2.0 / H2);

    for (int i = 0; i < H2; i++) {
        nn->W3[i] = rand_gaussian(0, std_w3);
    }
}

double forward(
    NeuralNetwork* nn,
    const double input[],
    double z1[],
    double a1[],
    double z2[],
    double a2[]
) {
    // Hidden Layer 1
    for (int i = 0; i < H1; i++) {
        z1[i] = nn->b1[i];

        for (int j = 0; j < NUM_FEATURES; j++) {
            z1[i] += nn->W1[i][j] * input[j];
        }

        a1[i] = relu(z1[i]);
    }

    // Hidden Layer 2
    for (int i = 0; i < H2; i++) {
        z2[i] = nn->b2[i];

        for (int j = 0; j < H1; j++) {
            z2[i] += nn->W2[i][j] * a1[j];
        }

        a2[i] = relu(z2[i]);
    }

    // Output Layer
    double z3 = nn->b3;

    for (int i = 0; i < H2; i++) {
        z3 += nn->W3[i] * a2[i];
    }

    return sigmoid(z3);
}
