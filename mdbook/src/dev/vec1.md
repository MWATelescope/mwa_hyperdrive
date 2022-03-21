# Non-empty vectors (`vec1`)

See the [official docs](https://docs.rs/vec1/latest/vec1/) for more info.

Why do we use `Vec1` instead of `Vec`? Can't we just assume that all our vectors
are not empty? Well, we could, but then:

1. We'd probably be wrong at some point in the future;

2. We're making the code inject panic code somewhere it probably doesn't need to; and

3. We can do better than this.

[This
article](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/),
although catered for Haskell, presents the case well. Put simply, if we require
something to be non-empty, then we can express that with a type, and this means
we don't need to re-validate its non-empty-ness after its creation.

We can also avoid using `Vec` by using `Option<Vec1>`; we still need to check
whether we have `Some` or `None`, but doing so is better than assuming a `Vec`
is non-empty.
