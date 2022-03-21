# The `hyperdrive` source list format

Coordinates are right ascension (RA) and declination, both with units of degrees
in the J2000 epoch. All frequencies are in Hz and all flux densities are in Jy.

All Gaussian and shapelet sizes are in arcsec, but their position angles are in
degrees. In an image space where RA increases from right to left (i.e. bigger RA
values are on the left), position angles rotate counter clockwise. A
position angle of 0 has the major axis aligned with the declination axis.

`hyperdrive`-style source lists can be read from and written to either the
[YAML](https://yaml.org/) or [JSON](https://www.json.org/json-en.html) file
formats (YAML preferred). Example Python code to read and write these files is
in the [examples
directory](https://github.com/MWATelescope/mwa_hyperdrive/tree/main/examples).

As most sky-models only include Stokes I, Stokes Q, U and V are not required to
be specified. If they are not specified, they are assumed to have values of 0.

~~~admonish example
The following are the contents of a valid YAML file. `super_sweet_source1` is a
single-component point source with a list-type flux density.
`super_sweet_source2` has two components: one Gaussian with a power law, and a
shapelet with a curved power law.

```yaml
---
super_sweet_source1:
  - ra: 10.0
    dec: -27.0
    comp_type: point
    flux_type:
      list:
        - freq: 150000000.0
          i: 10.0
        - freq: 170000000.0
          i: 5.0
          q: 1.0
          u: 2.0
          v: 3.0
super_sweet_source2:
  - ra: 0.0
    dec: -35.0
    comp_type:
      gaussian:
        maj: 20.0
        min: 10.0
        pa: 75.0
    flux_type:
      power_law:
        si: -0.8
        fd:
          freq: 170000000.0
          i: 5.0
          q: 1.0
          u: 2.0
          v: 3.0
  - ra: 155.0
    dec: -10.0
    comp_type:
      shapelet:
        maj: 20.0
        min: 10.0
        pa: 75.0
        coeffs:
          - n1: 0
            n2: 1
            value: 0.5
    flux_type:
      curved_power_law:
        si: -0.6
        fd:
          freq: 150000000.0
          i: 50.0
          q: 0.5
          u: 0.1
        q: 0.2
```
~~~
