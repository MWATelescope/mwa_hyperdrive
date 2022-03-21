# The `RTS` source list format

Coordinates are right ascension and declination, which have units of decimal
hours (i.e. 0 - 24) and degrees, respectively. All frequencies are in Hz, and
all flux densities are in Jy.

Gaussian and shapelet sizes are specified in arcminutes, whereas position angles
are in degrees. In an image space where RA increases from right to left (i.e.
bigger RA values are on the left), position angles rotate counter clockwise. A
position angle of 0 has the major axis aligned with the declination axis.

All flux densities are specified in the "list" style (i.e. power laws and curved
power laws are not supported).

Keywords like `SOURCE`, `COMPONENT`, `POINT` etc. must be at the start of a line
(i.e. no preceding space).

RTS sources always have a "base source", which can be thought of as a
non-optional component or the first component in a collection of components.

~~~admonish example
Taken from [srclists](https://github.com/JLBLine/srclists), file
`srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2.txt`.

Single-component point source:

```plaintext
SOURCE J161720+151943 16.2889374 15.32883
FREQ 80.0e+6 1.45351 0 0 0
FREQ 100.0e+6 1.23465 0 0 0
FREQ 120.0e+6 1.07389 0 0 0
FREQ 140.0e+6 0.95029 0 0 0
FREQ 160.0e+6 0.85205 0 0 0
FREQ 180.0e+6 0.77196 0 0 0
FREQ 200.0e+6 0.70533 0 0 0
FREQ 220.0e+6 0.64898 0 0 0
FREQ 240.0e+6 0.60069 0 0 0
ENDSOURCE
```

Two component Gaussian source:

```plaintext
SOURCE EXT035221-3330 3.8722900 -33.51040
FREQ 150.0e+6 0.34071 0 0 0
FREQ 170.0e+6 0.30189 0 0 0
FREQ 190.0e+6 0.27159 0 0 0
FREQ 210.0e+6 0.24726 0 0 0
GAUSSIAN 177.89089 1.419894937734689 0.9939397975299238
COMPONENT 3.87266 -33.52005
FREQ 150.0e+6 0.11400 0 0 0
FREQ 170.0e+6 0.10101 0 0 0
FREQ 190.0e+6 0.09087 0 0 0
FREQ 210.0e+6 0.08273 0 0 0
GAUSSIAN 2.17287 1.5198465761214996 0.9715267232520484
ENDCOMPONENT
ENDSOURCE
```

Single component shapelet source (truncated):

```plaintext
SOURCE FornaxA 3.3992560 -37.27733
FREQ 185.0e+6 209.81459 0 0 0
SHAPELET2 68.70984356 3.75 4.0
COEFF 0.0 0.0 0.099731291104
COEFF 0.0 1.0 0.002170910745
COEFF 0.0 2.0 0.078201040179
COEFF 0.0 3.0 0.000766942939
ENDSOURCE
```
~~~
