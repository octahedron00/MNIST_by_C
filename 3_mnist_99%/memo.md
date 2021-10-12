### About using more than one layer

: width(number of floor) will also increase, 

: and training time will grow exponentially.

I couldn't find how to build a better model, with two convolutional layers...

---

### One-floor-model actually worked 

[3_mnist_99%_MaxPooling.cpp](3_mnist_99%_MaxPooling.cpp) worked well,

in about 40 min(30 epochs), the error rate dropped under 1%.

---

### Should I try next one, in C?

One big lesson : **Use NumPy.**

Writing neural net codes like this on **C++** is **so painful.**

> Now I know how Artificial Neural Network works,
>
> how to train NNs with back propagation,
>
> the things affect speed of training,
> 
> and a lot.

---

### Now I'll try using Python, TensorFlow

See you soon!
