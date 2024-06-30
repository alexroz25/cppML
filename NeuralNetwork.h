//
// Created by Alex on 6/29/2024.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <iostream>

// 784 -> 16 -> 10 neural network
class NeuralNetwork {
public:
    const double LEARNING_RATE;
    int trainingElts; // number of training examples
    int inputSize; // size of each training example (784)
    int hiddenSize;
    int outputSize;
    std::vector<int> correctLabels;
    Matrix<double> input; // a1: 60000 x 784
    Matrix<double> z2;
    Matrix<double> hidden; // a2: 60000 x 16
    Matrix<double> hiddenDelta; // d2: 60000 x 16
    Matrix<double> z3;
    Matrix<double> output; // a3: 60000 x 10
    Matrix<double> outputDelta; // d3: 60000 x 10
    Matrix<double> derivCost; // 60000 x 10
    std::vector<double> cost; // 60000
    Matrix<double> w2; // 16 x 784 // weights from inputLayer to hiddenLayer
    Matrix<double> w3; // 10 x 16 // row = output neuron index, col = hidden neuron index
    std::vector<double> b2;
    std::vector<double> b3;
    std::mt19937* mt;

    NeuralNetwork(double LEARNING_RATE, int trainingElts, int inputSize, int hiddenSize, int outputSize, std::mt19937* mt)
          : LEARNING_RATE(LEARNING_RATE),
            trainingElts(trainingElts), 
            inputSize(inputSize), 
            hiddenSize(hiddenSize), 
            outputSize(outputSize), 
            input(Matrix<double>(trainingElts, inputSize)), 
            z2(Matrix<double>(trainingElts, hiddenSize)),
            hidden(Matrix<double>(trainingElts, hiddenSize)),
            hiddenDelta(Matrix<double>(trainingElts, hiddenSize)),
            z3(Matrix<double>(trainingElts, outputSize)),
            output(Matrix<double>(trainingElts, outputSize)),
            outputDelta(Matrix<double>(trainingElts, outputSize)),
            derivCost(Matrix<double>(trainingElts, outputSize)),
            cost(std::vector<double>(trainingElts, 0)),
            w2(Matrix<double>(hiddenSize, inputSize)), 
            w3(Matrix<double>(outputSize, hiddenSize)), 
            b2(std::vector<double>(hiddenSize, 0)),
            b3(std::vector<double>(outputSize, 0)),
            mt(mt) {
        correctLabels.reserve(trainingElts);

        double hiddenRange = 1.0 / sqrt(inputSize);
        std::uniform_real_distribution<double> hiddenDist(-hiddenRange, hiddenRange);
        // initialize hidden layer Neuron weights
        for (int i = 0; i < inputSize * hiddenSize; ++i) w2.matrix[i] = hiddenDist(*mt);

        double outputRange = 1.0 / sqrt(hiddenSize);
        std::uniform_real_distribution<double> outputDist(-outputRange, outputRange);
        // initialize output layer Neuron weights
        for (int i = 0; i < hiddenSize * outputSize; ++i) w3.matrix[i] = outputDist(*mt);
    }

    Matrix<double> ReLU(Matrix<double>& m) {
        Matrix<double> out = m;
        for (double& elt : out.matrix) if (elt < 0) elt = 0;
        return out;
    }

    Matrix<double> softmax(Matrix<double>& m) {
        Matrix<double> out = m;
        uint64_t rows = out.numRows;
        uint64_t cols = out.numCols;
        
        for (uint64_t r = 0; r < rows; ++r) {
            double denom = 0;
            for (uint64_t c = 0; c < cols; ++c) denom += exp(out.at(r, c));
            for (uint64_t c = 0; c < cols; ++c) out.at(r, c) = exp(out.at(r, c)) / denom;
        }
        
        return out;
    }

    void forward_propagate() {
        Matrix<double> inputt = input.transpose();
        z2 = w2.multiply(inputt).transpose().add_to_each_row(b2);

        hidden = ReLU(z2);

        Matrix<double> hiddent = hidden.transpose();
        z3 = w3.multiply(hiddent).transpose().add_to_each_row(b3);

        output = softmax(z3);
    }

    void calculate_cost() {
        for (int i = 0; i < trainingElts; ++i) {
            int correctLabel = correctLabels[i];
            double eltCost = 0;
            for (int pred = 0; pred < outputSize; ++pred) {
                double temp = (pred == correctLabel) ? 1 - output.at(i, pred) : -output.at(i, pred);
                derivCost.at(i, pred) = temp;
                eltCost += (temp * temp);
            }
            cost[i] = eltCost;
        }
    }

    void calculate_deltas() {
        // calculate deltas for output layer
        for (int r = 0; r < trainingElts; ++r) {
            for (int o = 0; o < outputSize; ++o) {
                outputDelta.at(r, o) = 2 * derivCost.at(r, o) * output.at(r, o) * (1 - output.at(r, o));
            }
        }

        // calculate deltas for hidden layer
        for (int r = 0; r < trainingElts; ++r) {
            for (int h = 0; h < hiddenSize; ++h) {
                
                for (int o = 0; o < outputSize; ++o) {
                    hiddenDelta.at(r, h) += w3.at(o, h) * outputDelta.at(r, o);
                }
                hiddenDelta.at(r, h) *= (hidden.at(r, h) * (1 - hidden.at(r, h)));
            }
        }
    }

    void gradient_descent() {
        Matrix<double> dw3(outputSize, hiddenSize);
        std::vector<double> db3(outputSize, 0);

        for (int r = 0; r < trainingElts; ++r) {
            for (int o = 0; o < outputSize; ++o) {
                for (int h = 0; h < hiddenSize; ++h) {
                    dw3.at(o, h) += (outputDelta.at(r, o) * hidden.at(r, h));
                }

                db3[o] += outputDelta.at(r, o);
            }
        }
        for (double& d : dw3.matrix) d /= trainingElts;
        for (double& d : db3) d /= trainingElts;

        Matrix<double> dw3LRadjust = dw3.multiply(LEARNING_RATE);
        w3 = w3.add(dw3LRadjust);

        for (double& db : db3) db *= LEARNING_RATE;
        for (int i = 0; i < outputSize; ++i) b3[i] += db3[i];



        Matrix<double> dw2(hiddenSize, inputSize);
        std::vector<double> db2(hiddenSize, 0);

        for (int r = 0; r < trainingElts; ++r) {
            for (int h = 0; h < hiddenSize; ++h) {
                for (int i = 0; i < inputSize; ++i) {
                    dw2.at(h, i) += (hiddenDelta.at(r, h) * input.at(r, i));
                }

                db2[h] += hiddenDelta.at(r, h);
            }
        }
        for (double& d : dw2.matrix) d /= trainingElts;
        for (double& d : db2) d /= trainingElts;

        Matrix<double> dw2LRadjust = dw2.multiply(LEARNING_RATE);
        w2 = w2.add(dw2LRadjust);

        for (double& db : db2) db *= LEARNING_RATE;
        for (int i = 0; i < hiddenSize; ++i) b2[i] += db2[i];
    }

    double calculate_average_cost() {
        double sum = 0;
        for (double& c : cost) {
            sum += c;
        }
        return sum / trainingElts;
    }

    double calculate_accuracy() {
        int correct = 0;
        for (int r = 0; r < trainingElts; ++r) {
            double rowMax = -1;
            int maxdex = -1;
            for (int pred = 0; pred < outputSize; ++pred) {
                if (output.at(r, pred) > rowMax) {
                    rowMax = output.at(r, pred);
                    maxdex = pred;
                }
            }
            if (maxdex == correctLabels[r]) ++correct;
        }

        return (double)correct / (double)trainingElts;
    }

    void read_csv(const std::string& filename) {
        std::fstream fin;
        fin.open(filename, std::ios::in);

        std::string elt;
        for (int row = 0; row < trainingElts; ++row) {
            // get label
            getline(fin, elt, ',');
            correctLabels.push_back(stoi(elt));

            // get pixels
            for (int pixel = 0; pixel < inputSize - 1; ++pixel) {
                getline(fin, elt, ',');
                input.at(row, pixel) = stoi(elt) / 255.0;
            }

            // edge case for getting the last element
            getline(fin, elt, '\n');
            input.at(row, inputSize - 1) = stoi(elt) / 255.0;
        }
        fin.close();
    }


};

#endif // NEURALNETWORK_H
