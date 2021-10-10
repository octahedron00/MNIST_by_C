## 2_mnist_CNN

### From 28\*28, Using Convolutional Neural Net floor.

Simple convolutional neural networks.

Training : 60000 images, Testing : 10000 images 

This code can get about **1.5% error rate in 10 epochs.**

---

### [2_mnist_CNN.cpp](2_mnist_CNN.cpp)

C1 : **28\*28 to 16@24\*24**, by 16 5\*5 convolution nets

no paddings, no strides

using **Sigmoid** as activation func

S2 : **16\*24\*24 to 32**, using **Sigmoid** as activation func

S3 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ 1.4% in 10 epochs

Features : **Better Accuracy**, because of adding positional information

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.
