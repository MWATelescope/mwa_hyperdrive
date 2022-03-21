# The André Offringa (`ao`) source list format

This format is used by `calibrate` within `mwa-reduce` (closed-source code).

RA is in decimal hours (0 to 24) and Dec is in degrees in the J2000 epoch, but
sexagesimal formatted. All frequencies and flux densities have their units
annotated (although these appear to only be MHz and Jy, respectively).

Point and Gaussian components are supported, but not shapelets. All Gaussian
sizes are in arcsec, but their position angles are in degrees. In an image space
where RA increases from right to left (i.e. bigger RA values are on the left),
position angles rotate counter clockwise. A
position angle of 0 has the major axis aligned with the declination axis.

Flux densities must be specified in the power law or "list" style (i.e. curved
power laws are not supported).

Source names are allowed to have spaces inside them, because the names are
surrounded by quotes. This is fine for reading, but when converting one of these
sources to another format, the spaces need to be translated to underscores.

~~~admonish example
```plaintext
skymodel fileformat 1.1
source {
  name "J002549-260211"
  component {
    type point
    position 0h25m49.2s -26d02m13s
    measurement {
      frequency 80 MHz
      fluxdensity Jy 15.83 0 0 0
    }
    measurement {
      frequency 100 MHz
      fluxdensity Jy 16.77 0 0 0
    }
  }
}
source {
  name "COM000338-1517"
  component {
    type gaussian
    position 0h03m38.7844s -15d17m09.7338s
    shape 89.05978540785397 61.79359416237104 89.07023307815388
    sed {
      frequency 160 MHz
      fluxdensity Jy 0.3276758375536325 0 0 0
      spectral-index { -0.9578697792073567 0.00 }
    }
  }
}
```
~~~
