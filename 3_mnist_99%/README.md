## 3_mnist_99%

### CNNs To get accuracy over 99%.

Convolutional neural networks.

Training : 60000 images, Testing : 10000 images 

This code can get about **0.9% error rate in 30 epochs.**

---

### [3_mnist_99%_MaxPooling.cpp](3_mnist_99%_MaxPooling.cpp)

> C1 : **28\*28 to 16@14\*14**, by 16 9\*9 convolution nets
>
> paddings 3(top and left) and 4(bottom and right), strides 2
> 
> using **Sigmoid** as activation func
>
> C2 : **16@14\*14 to 16@7\*7**, by 2\*2 max pooling
>
> S3 : **16\*7\*7 to 32**, using **Sigmoid** as activation func
>
> S4 : **32 to 10**, using **Sigmoid** as activation func

Error rate : ~ 1.5% in 10 epochs, **0.9%** in 30 epochs.

Total Training Time : about **12 min** for 10 epochs

Features : **Faster time and better accuracy**, maybe bigger convolutional net works better..

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.
