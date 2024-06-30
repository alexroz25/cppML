#include <iostream>
#include "NeuralNetwork.h"
using namespace std;

constexpr double LEARNING_RATE = 0.5;
constexpr int NUM_EPOCHS = 1000;

int main() {
    ios_base::sync_with_stdio(false);
    std::random_device rd;
    std::mt19937 mt(0); // Replace with 0 with rd() if you want a random seed

    NeuralNetwork train(LEARNING_RATE, 60000, 784, 16, 10, &mt);

    train.read_csv("mnist_train.csv");

    double currCost;
    double prevCost = 0;

    int n = 0;
    while (n < NUM_EPOCHS) {
        train.forward_propagate();
        train.calculate_cost();

        currCost = train.calculate_average_cost();
        cout << "Epoch: " << n++ << " | Accuracy: " << train.calculate_accuracy() << " | Average Cost: " << currCost << " | Improvement: " << prevCost - currCost << " | Learning Rate: " << train.LEARNING_RATE << endl;
        prevCost = currCost;

        train.calculate_deltas();
        train.gradient_descent();
    }
    // final epoch results
    train.forward_propagate();
    train.calculate_cost();
    currCost = train.calculate_average_cost();
    cout << "Epoch: " << n << " | Accuracy: " << train.calculate_accuracy() << " | Average Cost: " << currCost << " | Improvement: " << prevCost - currCost << endl;
    prevCost = currCost;

    // test weights and biases on new data
    NeuralNetwork test(LEARNING_RATE, 10000, 784, 16, 10, &mt);
    test.read_csv("mnist_test.csv");
    test.w2 = train.w2;
    test.w3 = train.w3;
    test.b2 = train.b2;
    test.b3 = train.b3;
    test.forward_propagate();
    cout << "Accuracy on mnist_test.csv after " << n << " epochs: " << test.calculate_accuracy() << endl;
    return 0;
}
