# MNIST_by_C

### Neural Networks solving MNIST only with C/C++.

#### Without any package, class, pre-written code.

#### Also Known As, "just use NumPy, please."

These can be used for education of neural network, since C++ can be easily read.

The codes are written by Octo Moon : <octahedron00@gmail.com>

---

## How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.

---

## [0_mnist_reader](0_mnist_reader/)

A simple file reader.

You can select intervals to read.

This code shows data in dataset by using own ASCII-art.

---

## [1_mnist_NN](1_mnist_NN/)

### 784 - 32 - 32 - 10

A simple neural network.

Training : 60000 images, Testing : 10000 images 

This code can get about **5% error rate in 10 epochs.**

---

## [2_mnist_CNN](2_mnist_CNN/)

### 28\*28 - 16@24\*24 - 32 - 10

A simple convolutional neural network.

Training : 60000 images, Testing : 10000 images 

This uses 5\*5 convolutional nets.

This code can get about **1.4% error rate in 10 epochs.**

---

## [3_mnist_99%](3_mnist_99%/)

### 35\*35 - 8@14\*14 - 8@7\*7 - 32 - 10

A CNN to make 99% accuracy faster.

Training : 60000 images, Testing : 10000 images 

This uses 9\*9 convolutional nets, 3 and 4 paddings, 2 strides.

This code can get about **0.9% error rate in "30" epochs.**
