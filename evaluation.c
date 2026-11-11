
#include <stdio.h>

#include "evaluation.h"

double find_best_threshold(
    NeuralNetwork* nn,
    SensorData val[],
    int val_size
) {
    double best_thresh = 0.40;
    double best_f1 = 0.0;

    for (double t = 0.35; t <= 0.45; t += 0.01) {

        int tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < val_size; i++) {

            double z1[H1], a1[H1];
            double z2[H2], a2[H2];

            double prob = forward(
                nn, val[i].features,
                z1, a1, z2, a2
            );

            int pred = (prob >= t) ? 1 : 0;
            int actual = val[i].target;

            if (actual == 1 && pred == 1) tp++;
            else if (actual == 0 && pred == 1) fp++;
            else if (actual == 1 && pred == 0) fn++;
        }

        double prec =
            (tp + fp > 0) ? (double)tp / (tp + fp) : 0;

        double rec =
            (tp + fn > 0) ? (double)tp / (tp + fn) : 0;

        double f1 =
            (prec + rec > 0)
            ? 2 * prec * rec / (prec + rec)
            : 0;

        if (f1 > best_f1) {
            best_f1 = f1;
            best_thresh = t;
        }
    }

    return best_thresh;
}

void evaluate_model(
    NeuralNetwork* nn,
    SensorData data[],
    int size,
    double threshold
) {
    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < size; i++) {

        double z1[H1], a1[H1];
        double z2[H2], a2[H2];

        double prob = forward(
            nn, data[i].features,
            z1, a1, z2, a2
        );

        int prediction = (prob >= threshold) ? 1 : 0;
        int actual = data[i].target;

        if (actual == 1 && prediction == 1) tp++;
        else if (actual == 0 && prediction == 1) fp++;
        else if (actual == 0 && prediction == 0) tn++;
        else if (actual == 1 && prediction == 0) fn++;
    }

    double accuracy = (double)(tp + tn) / size;

    double precision =
        (tp + fp > 0) ? (double)tp / (tp + fp) : 0;

    double recall =
        (tp + fn > 0) ? (double)tp / (tp + fn) : 0;

    double f1 =
        (precision + recall > 0)
        ? 2 * precision * recall / (precision + recall)
        : 0;

    printf("\n====================================\n");
    printf(" MODEL EVALUATION (Threshold = %.2f)\n", threshold);
    printf("====================================\n");

    printf("Confusion Matrix:\n");
    printf("             Predicted 0   Predicted 1\n");
    printf("Actual 0 : %12d   %12d\n", tn, fp);
    printf("Actual 1 : %12d   %12d\n\n", fn, tp);

    printf("Accuracy  : %.4f\n", accuracy);
    printf("Precision : %.4f\n", precision);
    printf("Recall    : %.4f\n", recall);
    printf("F1-Score  : %.4f\n", f1);
}
