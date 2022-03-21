# Multiple-dimension arrays (`ndarray`)

[`ndarray`](https://docs.rs/ndarray/latest/ndarray/)s are used liberally
throughout `hyperdrive` (and its dependencies). `ndarray`'s usage is a little
different to the usual Rust vectors and slices. This page hopes to help
developers understand what some of the loops using `ndarray`s is doing.

Here's a simplified example:

```rust,ignore
use marlu::Jones;
use ndarray::Array3;

// Set up `vis` and `weights` to be 3D arrays. The dimensions are time,
// baseline, channel.
let shape = (2, 8128, 768);
let mut vis: Array3<Jones<f32>> = Array3::from_elem(shape, Jones::identity());
let mut weights: Array3<f32> = Array3::ones(shape);
// `outer_iter_mut` mutably iterates over the slowest dimension (in this
// case, time).
vis.outer_iter_mut()
    // Iterate over weights at the same time as `vis`.
    .zip(weights.outer_iter_mut())
    // Also get an index of how far we are into the time dimension.
    .enumerate()
    .for_each(|(i_time, (mut vis, mut weights))| {
        // `vis` and `weights` are now 2D arrays. `i_time` starts from 0 and
        // is an index for the time dimension.
        vis.outer_iter_mut()
            .zip(weights.outer_iter_mut())
            .enumerate()
            .for_each(|(i_bl, (mut vis, mut weights))| {
                // `vis` and `weights` are now 1D arrays. `i_bl` starts from
                // 0 and is an index for the baseline dimension.

                // Use standard Rust iterators to get the
                // elements of the 1D arrays.
                vis.iter_mut().zip(weights.iter_mut()).enumerate().for_each(
                    |(i_chan, (vis, weight))| {
                        // `vis` is a mutable references to a Jones matrix
                        // and `weight` is a mutable reference to a float.
                        // `i_chan` starts from 0 and is an index for the
                        // channel dimension.
                        // Do something with these references.
                        *vis += Jones::identity() * (i_time + i_bl + i_chan) as f32;
                        *weight += 2.0;
                    },
                );
            });
    });
```

## Views

It is typical to pass Rust `Vec`s around as slices, i.e. a `Vec<f64>` is
borrowed as a `&[f64]`. Similarly, one might be tempted to make a function
argument a borrowed `ndarray`, e.g. `&Array3<f64>`, but there is a better way.
Calling `.view()` or `.view_mut()` on an `ndarray` yields an `ArrayView` or
`ArrayViewMut`, which can be any subsection of the full array. By using views we
can avoid requiring a borrow of the whole array when we only want a part of it.
