
#include <stdio.h>
#include <math.h>

#include "training.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double weighted_bce_loss(double y, double y_hat) {
    if (y_hat < 1e-15) y_hat = 1e-15;
    if (y_hat > 1 - 1e-15) y_hat = 1 - 1e-15;

    return -(POS_WEIGHT * y * log(y_hat) +
             (1.0 - y) * log(1.0 - y_hat));
}

void train_network(
    NeuralNetwork* nn,
    SensorData train[],
    int train_size
) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        shuffle_dataset(train, train_size);

        double total_loss = 0;

        double current_lr =
            0.0001 +
            0.5 * (LEARNING_RATE - 0.0001) *
            (1.0 + cos(M_PI * epoch / EPOCHS));

        for (int i = 0; i < train_size; i += BATCH_SIZE) {

            int current_batch_size =
                (i + BATCH_SIZE > train_size)
                ? (train_size - i)
                : BATCH_SIZE;

            double dW1_acc[H1][NUM_FEATURES] = {0};
            double db1_acc[H1] = {0};

            double dW2_acc[H2][H1] = {0};
            double db2_acc[H2] = {0};

            double dW3_acc[H2] = {0};
            double db3_acc = 0;

            for (int b = 0; b < current_batch_size; b++) {

                int sample_idx = i + b;

                double z1[H1], a1[H1];
                double z2[H2], a2[H2];

                double y_hat = forward(
                    nn,
                    train[sample_idx].features,
                    z1, a1, z2, a2
                );

                double y = train[sample_idx].target;

                total_loss += weighted_bce_loss(y, y_hat);

                // Output Layer
                double dz3 =
                    y_hat * (1.0 + y * (POS_WEIGHT - 1.0))
                    - POS_WEIGHT * y;

                for (int j = 0; j < H2; j++) {
                    dW3_acc[j] += dz3 * a2[j];
                }

                db3_acc += dz3;

                // Hidden Layer 2
                double dz2[H2];

                for (int j = 0; j < H2; j++) {

                    double da2 = dz3 * nn->W3[j];

                    dz2[j] =
                        da2 * relu_derivative(z2[j]);

                    for (int k = 0; k < H1; k++) {
                        dW2_acc[j][k] += dz2[j] * a1[k];
                    }

                    db2_acc[j] += dz2[j];
                }

                // Hidden Layer 1
                double dz1[H1];

                for (int j = 0; j < H1; j++) {

                    double da1 = 0;

                    for (int k = 0; k < H2; k++) {
                        da1 += dz2[k] * nn->W2[k][j];
                    }

                    dz1[j] =
                        da1 * relu_derivative(z1[j]);

                    for (int k = 0; k < NUM_FEATURES; k++) {
                        dW1_acc[j][k] +=
                            dz1[j] *
                            train[sample_idx].features[k];
                    }

                    db1_acc[j] += dz1[j];
                }
            }

            // Update Layer 3
            for (int j = 0; j < H2; j++) {

                double g =
                    dW3_acc[j] / current_batch_size +
                    L2_LAMBDA * nn->W3[j];

                nn->vW3[j] =
                    MOMENTUM * nn->vW3[j] + current_lr * g;

                nn->W3[j] -= nn->vW3[j];
            }

            nn->vb3 =
                MOMENTUM * nn->vb3 +
                current_lr * (db3_acc / current_batch_size);

            nn->b3 -= nn->vb3;

            // Update Layer 2
            for (int j = 0; j < H2; j++) {

                for (int k = 0; k < H1; k++) {

                    double g =
                        dW2_acc[j][k] / current_batch_size +
                        L2_LAMBDA * nn->W2[j][k];

                    nn->vW2[j][k] =
                        MOMENTUM * nn->vW2[j][k] +
                        current_lr * g;

                    nn->W2[j][k] -= nn->vW2[j][k];
                }

                nn->vb2[j] =
                    MOMENTUM * nn->vb2[j] +
                    current_lr * (db2_acc[j] / current_batch_size);

                nn->b2[j] -= nn->vb2[j];
            }

            // Update Layer 1
            for (int j = 0; j < H1; j++) {

                for (int k = 0; k < NUM_FEATURES; k++) {

                    double g =
                        dW1_acc[j][k] / current_batch_size +
                        L2_LAMBDA * nn->W1[j][k];

                    nn->vW1[j][k] =
                        MOMENTUM * nn->vW1[j][k] +
                        current_lr * g;

                    nn->W1[j][k] -= nn->vW1[j][k];
                }

                nn->vb1[j] =
                    MOMENTUM * nn->vb1[j] +
                    current_lr * (db1_acc[j] / current_batch_size);

                nn->b1[j] -= nn->vb1[j];
            }
        }

        if (epoch % 50 == 0 || epoch == EPOCHS - 1) {
            printf(
                "Epoch %3d | LR = %.6f | Weighted Loss = %.6f\n",
                epoch,
                current_lr,
                total_loss / train_size
            );
        }
    }
}
