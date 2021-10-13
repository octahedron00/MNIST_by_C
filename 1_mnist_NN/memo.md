### Now It Really Worked!

The only one purpose of these codes is :

: to test whether I truely understand how AI and gradient descent works.

Some of students who use TensorFlow or else, don't know how gradient descent really works...

So, I tried to make MNIST-solving Neural Net,

but **without TensorFlow, NumPy, only with C/C++**.

---

### The code is really dirty, I know...

These codes have **no defined testing functions, matrix classes or operators.**

I wrote them this way, because **it looks more intuitive, *maybe just for me...***

> If you want to learn from this code, you can just read down the whole code,
>
> without jumping to any positions where some functions are declared.
>
> For me, again, these codes helped me to understand how AI works.

And I didn't want to change the form of codes...

> I wrote them all, and now I want to try Python.
>
> ~~I just don't want to work again with C...~~

---

### Speed problem : ReLu

: Since sigmoid takes a lot of calculation(exponentional function)...

: So, ReLu was suggested when someone found that sigmoid doesn't work well.

But, how to make ReLu in C?

#### Using Softmax as activation function to normalize results.

So, I first tried to make softmax working in my code.

After a few of debuggings, It worked.

#### Just putting ReLu doesn't work well...

Now, I'm handling with it.
