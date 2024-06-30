#include <iostream>
#include "NeuralNetwork.h"
using namespace std;

constexpr double LEARNING_RATE = 0.5;

int main() {
    ios_base::sync_with_stdio(false);
    std::random_device rd;
    std::mt19937 mt(rd());

    NeuralNetwork NN(LEARNING_RATE, 60000, 784, 16, 10, &mt);

    NN.read_csv("E:/CLionProjects/cpp_ml_works/mnist_train.csv");

    NN.forward_propagate();
    NN.calculate_cost();
    double currCost = NN.calculate_average_cost();
    cout << "Epoch: 0 | Accuracy: " << NN.calculate_accuracy() << " | Average Cost: " << currCost << endl;
    double prevCost = currCost;
    int n = 0;
    while (n++ < 1000) {
        //cout << "forward_propagate()" << endl;
        NN.forward_propagate();

        //cout << "calculate_cost()" << endl;
        NN.calculate_cost();

        //cout << "calculate_deltas()" << endl;
        NN.calculate_deltas();

        //cout << "gradient_descent()" << endl;
        NN.gradient_descent();

        currCost = NN.calculate_average_cost();
        cout << "Epoch: " << n << " | Accuracy: " << NN.calculate_accuracy() << " | Average Cost: " << currCost << " | Improvement: " << prevCost - currCost << endl;
        prevCost = currCost;
    }

    NeuralNetwork test(LEARNING_RATE, 10000, 784, 16, 10, &mt);
    test.w2 = NN.w2;
    test.w3 = NN.w3;
    test.b2 = NN.b2;
    test.b3 = NN.b3;
    NN.forward_propagate();
    cout << "Accuracy on mnist_test.csv after " << n << " epochs: " << NN.calculate_accuracy() << endl;
    return 0;
}
