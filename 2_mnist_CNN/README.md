## 2_mnist_CNN

### From 28\*28, Using Convolutional Neural Net floor.

Simple convolutional neural networks.

Training : 60000 images, Testing : 10000 images 

This code can get about **1.5% error rate in 10 epochs.**

**[My memo here](memo.md)**

---

### [2_mnist_CNN.cpp](2_mnist_CNN.cpp)

> C1 : **28\*28 to 16@24\*24**, by 16 5\*5 convolution nets
>
> no paddings, no strides
> 
> using **Sigmoid** as activation func
>
> S2 : **16\*24\*24 to 32**, using **Sigmoid** as activation func
>
> S3 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ **1.4%** in 10 epochs

Total Training Time : about **70 min** for 10 epochs

Features : **Better Accuracy**, because of adding positional information by convolution

---

### [2_mnist_CNN_MaxPooling.cpp](2_mnist_CNN_MaxPooling.cpp)

> C1 : **28\*28 to 14\*14**, by 2\*2 max pooling
> 
> C2 : **14\*14 to 16@10\*10**, by 16 5\*5 convolution nets
>
> no paddings, no strides
> 
> using **Sigmoid** as activation func
>
> S3 : **16\*10\*10 to 32**, using **Sigmoid** as activation func
>
> S4 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ **2.8%** in 10 epochs

Total Training Time : about **10 min** for 10 epochs

Features : **Faster time**, because of less data using

---

### [2_mnist_CNN_Stride.cpp](2_mnist_CNN_Stride.cpp)

> C1 : **28\*28 to 16@13\*13**, by 16 5\*5 convolution nets
>
> no paddings, **strides 2**
> 
> using **Sigmoid** as activation func
>
> S2 : **16\*13\*13 to 32**, using **Sigmoid** as activation func
>
> S3 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ **1.6%** in 10 epochs

Total Training Time : about **20 min** for 10 epochs

Features : **Faster time**, because of using strides.

---

### [2_mnist_CNN_9by9.cpp](2_mnist_CNN_9by9.cpp)

> C1 : **28\*28 to 16@14\*14**, by 16 9\*9 convolution nets
>
> paddings 3(top and left) and 4(bottom and right), strides 2
> 
> using **Sigmoid** as activation func
>
> S2 : **16\*14\*14 to 32**, using **Sigmoid** as activation func
>
> S3 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ **1.5%** in 10 epochs

Total Training Time : about **45 min** for 10 epochs

Features : **Faster time and better accuracy**, maybe bigger convolutional net works better..

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.
