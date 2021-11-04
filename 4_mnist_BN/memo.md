### Batch Normalization

I calculated how its derivatives look like, and used that values.

It now worked!

---

### Not So Fast

For some Batch Normalization in layer of size n,

The calculation time complexity(TC) is about **O(n^2)**.

Like, **no other calculation takes that much time.**

(TC of making m-sized layer from n-sized layer by using whole connection is O(nm).)

That's why **if we use BN to a large layer, the learning takes a lot of time.**

---

### Why weight value matters?

When r goes bigger, like when r is 0.5, the all values go overflow.

Still don't know why... There must be some reason.

> Since we can't use bigger weight, 
>
> **the interval of initial random value now matters to speed of training.**
