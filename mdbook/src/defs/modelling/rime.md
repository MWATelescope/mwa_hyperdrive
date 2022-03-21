# Measurement equation

~~~admonish
A lot of this content was taken from [Jack Line's
`WODEN`](https://github.com/JLBLine/WODEN).
~~~

The measurement equation (also known as the Radio Interferometer Measurement
Equation; RIME) used in `hyperdrive`'s calculations is:

\\[ V_{s,f}(u_f,v_f,w_f) = \int\int S_{s,f}(l,m) e^{2 \pi i \phi} \frac{dl dm}{n} \\]

where
- \\( V_{s,f}(u_f,v_f,w_f) \\) is the measured visibility in some Stokes
  polarisation \\( s \\) for some frequency \\( f \\) at baseline coordinates
  \\( u_f, v_f, w_f \\);
- \\( S_{s,f} \\) is the apparent brightness in the direction \\( l, m \\) at
  the same frequency \\( f \\);
- \\( i \\) is the imaginary unit;
- \\( \phi = \left(u_fl + v_fm + w_f(n-1) \right) \\); and
- \\( n = \sqrt{1 - l^2 - m^2} \\).

As we cannot ever have the true \\( S_{s,f} \\) function, we approximate with a
sky-model source list, which details the expected positions and brightnesses of
sources. This effectively turns the above continuous equation into a discrete
one:

\\[ V_{s,f}(u_f,v_f,w_f) = \sum S_{s,f}(l,m) e^{2 \pi i \left(u_fl + v_fm + w_f(n-1) \right)} \\]

`hyperdrive` implements this equation as code, either on the CPU or GPU
(preferred), and it is a good example of an embarrassingly parallel problem.
