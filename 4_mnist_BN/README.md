## 4_mnist_BN

### Usage of ReLU and Batch Normalization

NN and CNN, for both.

Training : 60000 images, Testing : 10000 images 

This code can get about **3.5% error rate in 30 epochs.**

**[My memo Here](memo.md)**

---

### [4_mnist_BN_ReLU.cpp](4_mnist_BN_ReLU.cpp)

> S1 : **784 to 32**, using **Leak-ReLU** as activation func
>
> B1 : **Batch Normalization** of S1, 32 to 32
>
> S2 : **32 to 32**, using **Leak-ReLU** as activation func
>
> B2 : **Batch Normalization** of S2, 32 to 32
>
> S3 : **32 to 10**, using **Leak-ReLU** as activation func
>
> B2 : **Batch Normalization** of S3, 10 to 10

Error rate : ~ 1.5% in 10 epochs, **3.5%** in 10 epochs.

Total Training Time : about **3 min** for 10 epochs

Features : **Now it works with ReLU!** But not faster than Sigmoid NN, because of BN.

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.
