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

### [4_mnist_BN_MPCNN.cpp](4_mnist_BN_MPCNN.cpp)

Same shape with 3_mnist_99%_MaxPooling.cpp

> C1 : **28\*28 to 16@14\*14**, by 16 9\*9 convolution nets
>
> paddings 3(top and left) and 4(bottom and right), strides 2
> 
> using **Leak-ReLU** as activation func
>
> C2 : **16@14\*14 to 16@7\*7**, by 2\*2 max pooling
> 
> B2 : **Batch Normalization** of whole C2, 16@7\*7 to 16@7\*7
> 
> S3 : **16\*7\*7 to 32**, using **Leak-ReLU** as activation func
> 
> B3 : **Batch Normalization** of whole S3, 32 to 32
>
> S4 : **32 to 10**, using **Leak-ReLU** as activation func
> 
> B4 : **Batch Normalization** of whole S4, 10 to 10

Error rate : ~ 1.5% in 10 epochs

Total Training Time : about **40 min** for 10 epochs

Features : **...?** Actually this code is about 3 times slower than prior code. Proof of that BN with CNN is also avalible.

---

### How to use

You can download exe files to run program, or cpp files to read and edit.

To use programs, you should **download MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/)**, and put them **in the same place** with exe file.
