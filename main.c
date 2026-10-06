

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


// 1. CẤU HÌNH HẰNG SỐ VÀ HYPERPARAMETERS TỐI ƯU

#define MAX_SAMPLES  10000     // Số mẫu tối đa đọc từ CSV
#define NUM_FEATURES 7         // 5 features gốc + 2 features dẫn xuất vật lý

// Kiến trúc Mạng
#define H1 64              // Số node lớp ẩn 1 (Hidden Layer 1)
#define H2 32               // Số node lớp ẩn 2 (Hidden Layer 2)

#define EPOCHS     500         // Số chu kỳ huấn luyện
#define BATCH_SIZE 32          // Kích thước Mini-batch


// Tối ưu hóa siêu tham số
#define LEARNING_RATE 0.0012   // Tốc độ học ban đầu
#define MOMENTUM      0.90     // Quán tính SGD Momentum
#define POS_WEIGHT    2.20     // Trọng số phạt lớp lỗi
#define L2_LAMBDA     0.00015  // Hệ số Regularization L2


// 2. CẤU TRÚC DỮ LIỆU

typedef struct {
    double raw_features[5];     // Dữ liệu cảm biến gốc từ file CSV
    double features[NUM_FEATURES]; // Dữ liệu sau Feature Engineering & Scaling
    int target;
} SensorData;

typedef struct {
    // Trọng số và Độ lệch
    double W1[H1][NUM_FEATURES], b1[H1];
    double W2[H2][H1],           b2[H2];
    double W3[H2],               b3;

    // Biến lưu vận tốc quán tính (Momentum Velocity)
    double vW1[H1][NUM_FEATURES], vb1[H1];
    double vW2[H2][H1],           vb2[H2];
    double vW3[H2],               vb3;
} NeuralNetwork;


// 3. XỬ LÝ DỮ LIỆU (DATA PROCESSING & FEATURE ENGINEERING)


// Đọc file CSV dữ liệu cảm biến
int load_csv(const char* filename, SensorData dataset[]) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Loi: Khong the mo file '%s'!\n", filename);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file); // Bỏ qua dòng Header

    int count = 0;
    while (fgets(line, sizeof(line), file) && count < MAX_SAMPLES) {
        line[strcspn(line, "\r\n")] = 0;
        char* token = strtok(line, ",");
        if (!token) continue;

        strtok(NULL, ","); // Bỏ qua UDI
        strtok(NULL, ","); // Bỏ qua Product ID

        int valid = 1;
        for (int i = 0; i < 5; i++) {
            token = strtok(NULL, ",");
            if (!token) { valid = 0; break; }
            dataset[count].raw_features[i] = atof(token);
        }

        if (!valid) continue;

        token = strtok(NULL, ",");
        if (token) {
            dataset[count].target = atoi(token);
            count++;
        }
    }

    fclose(file);
    return count;
}

// Tráo ngẫu nhiên tập dữ liệu (Fisher-Yates Shuffle)
void shuffle_dataset(SensorData dataset[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        SensorData temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }
}

// Biến đổi đặc trưng vật lý & Chuẩn hóa dữ liệu (Standardization - Z-Score)

void feature_engineering_and_scaler(SensorData train[], int train_size,
                                   SensorData val[], int val_size,
                                   SensorData test[], int test_size) {
    SensorData* all_data[3] = {train, val, test};
    int sizes[3] = {train_size, val_size, test_size};

    // Feature Engineering
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < sizes[s]; i++) {
            for (int j = 0; j < 5; j++) {
                all_data[s][i].features[j] = all_data[s][i].raw_features[j];
            }
            // Feature 5: Chênh lệch nhiệt độ (Process Temp - Air Temp)
            all_data[s][i].features[5] = all_data[s][i].raw_features[1] - all_data[s][i].raw_features[0];
            // Feature 6: Công suất cơ học tương đối (Rotational Speed * Torque)
            all_data[s][i].features[6] = all_data[s][i].raw_features[2] * all_data[s][i].raw_features[3];
        }
    }

    // Tính Mean & Std trên duy nhất tập TRAIN (Tránh Data Leakage)
    double means[NUM_FEATURES] = {0}, stds[NUM_FEATURES] = {0};

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) means[j] += train[i].features[j];
    }
    for (int j = 0; j < NUM_FEATURES; j++) means[j] /= train_size;

    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            double diff = train[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < NUM_FEATURES; j++) stds[j] = sqrt(stds[j] / train_size);

    // Chuẩn hóa Z-score cho cả 3 tập Train, Val, Test theo thông số của Train
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < sizes[s]; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                if (stds[j] != 0) {
                    all_data[s][i].features[j] = (all_data[s][i].features[j] - means[j]) / stds[j];
                }
            }
        }
    }
}

 // 4. HÀM KÍCH HOẠT VÀ KHỞI TẠO MẠNG (ACTIVATIONS & INITIALIZATION)

double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

double sigmoid(double x) {
    if (x < -50) return 0;
    if (x > 50) return 1;
    return 1.0 / (1.0 + exp(-x));
}

// Tạo phân phối chuẩn Gauss cho He Initialization
double rand_gaussian(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}

void initialize_network(NeuralNetwork* nn) {
    memset(nn, 0, sizeof(NeuralNetwork));

    // He Initialization tối ưu cho hàm kích hoạt ReLU
    double std_w1 = sqrt(2.0 / NUM_FEATURES);
    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) nn->W1[i][j] = rand_gaussian(0, std_w1);
    }

    double std_w2 = sqrt(2.0 / H1);
    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H1; j++) nn->W2[i][j] = rand_gaussian(0, std_w2);
    }

    double std_w3 = sqrt(2.0 / H2);
    for (int i = 0; i < H2; i++) nn->W3[i] = rand_gaussian(0, std_w3);
}


 // 5. LAN TRUYỀN TIẾN VÀ HÀM MẤT MÁT

double forward(NeuralNetwork* nn, const double input[], double z1[], double a1[], double z2[], double a2[]) {
    // Hidden Layer 1 (ReLU)
    for (int i = 0; i < H1; i++) {
        z1[i] = nn->b1[i];
        for (int j = 0; j < NUM_FEATURES; j++) z1[i] += nn->W1[i][j] * input[j];
        a1[i] = relu(z1[i]);
    }

    // Hidden Layer 2 (ReLU)
    for (int i = 0; i < H2; i++) {
        z2[i] = nn->b2[i];
        for (int j = 0; j < H1; j++) z2[i] += nn->W2[i][j] * a1[j];
        a2[i] = relu(z2[i]);
    }

    // Output Layer (Sigmoid)
    double z3 = nn->b3;
    for (int i = 0; i < H2; i++) z3 += nn->W3[i] * a2[i];

    return sigmoid(z3);
}

// Weighted Binary Cross-Entropy Loss (Phạt nặng lỗi bỏ sót máy hỏng)
double weighted_bce_loss(double y, double y_hat) {
    if (y_hat < 1e-15) y_hat = 1e-15;
    if (y_hat > 1 - 1e-15) y_hat = 1 - 1e-15;
    return -(POS_WEIGHT * y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat));
}

 // 6. HUẤN LUYỆN MẠNG

void train_network(NeuralNetwork* nn, SensorData train[], int train_size) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_dataset(train, train_size);
        double total_loss = 0;

        // Lịch Learning Rate Decay
        double current_lr = 0.0001 + 0.5 * (LEARNING_RATE - 0.0001) * (1.0 + cos(M_PI * epoch / EPOCHS));

        for (int i = 0; i < train_size; i += BATCH_SIZE) {
            int current_batch_size = (i + BATCH_SIZE > train_size) ? (train_size - i) : BATCH_SIZE;

            // Biến tích lũy Gradient cho Mini-batch
            double dW1_acc[H1][NUM_FEATURES] = {0}, db1_acc[H1] = {0};
            double dW2_acc[H2][H1] = {0},           db2_acc[H2] = {0};
            double dW3_acc[H2] = {0},               db3_acc = 0;

            for (int b = 0; b < current_batch_size; b++) {
                int sample_idx = i + b;
                double z1[H1], a1[H1], z2[H2], a2[H2];

                double y_hat = forward(nn, train[sample_idx].features, z1, a1, z2, a2);
                double y = train[sample_idx].target;

                total_loss += weighted_bce_loss(y, y_hat);

                // --- Lan truyền ngược (Backpropagation) ---
                // Gradient Layer 3
                double dz3 = y_hat * (1.0 + y * (POS_WEIGHT - 1.0)) - POS_WEIGHT * y;
                for (int j = 0; j < H2; j++) dW3_acc[j] += dz3 * a2[j];
                db3_acc += dz3;

                // Gradient Layer 2
                double dz2[H2];
                for (int j = 0; j < H2; j++) {
                    double da2 = dz3 * nn->W3[j];
                    dz2[j] = da2 * relu_derivative(z2[j]);
                    for (int k = 0; k < H1; k++) dW2_acc[j][k] += dz2[j] * a1[k];
                    db2_acc[j] += dz2[j];
                }

                // Gradient Layer 1
                double dz1[H1];
                for (int j = 0; j < H1; j++) {
                    double da1 = 0;
                    for (int k = 0; k < H2; k++) da1 += dz2[k] * nn->W2[k][j];
                    dz1[j] = da1 * relu_derivative(z1[j]);
                    for (int k = 0; k < NUM_FEATURES; k++) dW1_acc[j][k] += dz1[j] * train[sample_idx].features[k];
                    db1_acc[j] += dz1[j];
                }
            }

            // --- Cập nhật trọng số với SGD Momentum & L2 Regularization ---
            // Update Layer 3
            for (int j = 0; j < H2; j++) {
                double g = dW3_acc[j] / current_batch_size + L2_LAMBDA * nn->W3[j];
                nn->vW3[j] = MOMENTUM * nn->vW3[j] + current_lr * g;
                nn->W3[j] -= nn->vW3[j];
            }
            nn->vb3 = MOMENTUM * nn->vb3 + current_lr * (db3_acc / current_batch_size);
            nn->b3 -= nn->vb3;

            // Update Layer 2
            for (int j = 0; j < H2; j++) {
                for (int k = 0; k < H1; k++) {
                    double g = dW2_acc[j][k] / current_batch_size + L2_LAMBDA * nn->W2[j][k];
                    nn->vW2[j][k] = MOMENTUM * nn->vW2[j][k] + current_lr * g;
                    nn->W2[j][k] -= nn->vW2[j][k];
                }
                nn->vb2[j] = MOMENTUM * nn->vb2[j] + current_lr * (db2_acc[j] / current_batch_size);
                nn->b2[j] -= nn->vb2[j];
            }

            // Update Layer 1
            for (int j = 0; j < H1; j++) {
                for (int k = 0; k < NUM_FEATURES; k++) {
                    double g = dW1_acc[j][k] / current_batch_size + L2_LAMBDA * nn->W1[j][k];
                    nn->vW1[j][k] = MOMENTUM * nn->vW1[j][k] + current_lr * g;
                    nn->W1[j][k] -= nn->vW1[j][k];
                }
                nn->vb1[j] = MOMENTUM * nn->vb1[j] + current_lr * (db1_acc[j] / current_batch_size);
                nn->b1[j] -= nn->vb1[j];
            }
        }

        // In tiến trình huấn luyện sau mỗi 50 epoch
        if (epoch % 50 == 0 || epoch == EPOCHS - 1) {
            printf("Epoch %3d | LR = %.6f | Weighted Loss = %.6f\n", epoch, current_lr, total_loss / train_size);
        }
    }
}


 // 7. TỐI ƯU NGƯỠNG DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH (TUNING & EVALUATION)

double find_best_threshold(NeuralNetwork* nn, SensorData val[], int val_size) {
    double best_thresh = 0.40;
    double best_f1 = 0.0;

    for (double t = 0.35; t <= 0.45; t += 0.01) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < val_size; i++) {
            double z1[H1], a1[H1], z2[H2], a2[H2];
            double prob = forward(nn, val[i].features, z1, a1, z2, a2);
            int pred = (prob >= t) ? 1 : 0;
            int actual = val[i].target;

            if (actual == 1 && pred == 1) tp++;
            else if (actual == 0 && pred == 1) fp++;
            else if (actual == 1 && pred == 0) fn++;
        }
        double prec = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
        double rec = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
        double f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0;

        if (f1 > best_f1) {
            best_f1 = f1;
            best_thresh = t;
        }
    }
    return best_thresh;
}

// Đánh giá mô hình
void evaluate_model(NeuralNetwork* nn, SensorData data[], int size, double threshold) {
    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < size; i++) {
        double z1[H1], a1[H1], z2[H2], a2[H2];
        double prob = forward(nn, data[i].features, z1, a1, z2, a2);
        int prediction = (prob >= threshold) ? 1 : 0;
        int actual = data[i].target;

        if (actual == 1 && prediction == 1) tp++;
        else if (actual == 0 && prediction == 1) fp++;
        else if (actual == 0 && prediction == 0) tn++;
        else if (actual == 1 && prediction == 0) fn++;
    }

    double accuracy  = (double)(tp + tn) / size;
    double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
    double recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
    double f1        = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;

    printf("\n====================================\n");
    printf("         MODEL EVALUATION (Threshold = %.2f)\n", threshold);
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


 // 8. HÀM MAIN

int main() {
    srand(42);

    SensorData* dataset = malloc(sizeof(SensorData) * MAX_SAMPLES);
    int total_samples = load_csv("ai4i2020.csv", dataset);
    printf("Da tai %d dong du lieu (7 Features | Seed=42 | Architecture=64-32 + Momentum + Cosine Decay)\n", total_samples);

    shuffle_dataset(dataset, total_samples);

    // Chia tập dữ liệu: Train 70%, Validation 15%, Test 15%
    int train_size = (int)(total_samples * 0.70);
    int val_size   = (int)(total_samples * 0.15);
    int test_size  = total_samples - train_size - val_size;

    SensorData* train_data = malloc(sizeof(SensorData) * train_size);
    SensorData* val_data   = malloc(sizeof(SensorData) * val_size);
    SensorData* test_data  = malloc(sizeof(SensorData) * test_size);

    for (int i = 0; i < train_size; i++) train_data[i] = dataset[i];
    for (int i = 0; i < val_size; i++)   val_data[i]   = dataset[train_size + i];
    for (int i = 0; i < test_size; i++)  test_data[i]  = dataset[train_size + val_size + i];

    feature_engineering_and_scaler(train_data, train_size, val_data, val_size, test_data, test_size);

    NeuralNetwork nn;
    initialize_network(&nn);

    printf("\n========== TRAINING ==========\n");
    train_network(&nn, train_data, train_size);

    printf("\n========== VALIDATION ==========\n");
    double best_threshold = find_best_threshold(&nn, val_data, val_size);
    evaluate_model(&nn, val_data, val_size, best_threshold);

    printf("\n========== TEST ==========\n");
    evaluate_model(&nn, test_data, test_size, best_threshold);


    free(dataset);
    free(train_data);
    free(val_data);
    free(test_data);

    return 0;
}