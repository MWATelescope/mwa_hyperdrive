# FITS source list formats

There are three supported fits file formats:
- LoBES: used in LoBES catalogue <https://doi.org/10.1017/pasa.2021.50>
- Jack: extended LoBES format for Jack Line's sourcelist repository, <https://github.com/JLBLine/srclists/>.
- Gleam: used in GLEAM-X pipeline <https://github.com/GLEAM-X/GLEAM-X-pipeline/tree/master/models>

These formats differ mostly in the names of columns, and component and flux types
supported. *LoBES* fits files support point, and Gaussian components with list,
power law and curved power law flux density models. *Jack* fits files extend the
LoBES format with an additional table for shapelet coefficients. *Gleam* fits are
similar to LoBES fits, but with different column names, and combine power law and
curved power law flux density models into a just two columns.

More info from [woden docs](https://woden.readthedocs.io/en/latest/operating_principles/skymodel.html)

## Source posititons

Coordinates are right ascension (RA) and declination, both with units of degrees
in the J2000 epoch. All frequencies are in Hz and all flux densities are in Jy.

Jack and LoBES fits formats use the columns `RA` and `DEC` for source positions,
while Gleam fits files use `RAJ2000` and `DEJ2000`.

## Component types

Jack and LoBES fits formats use the column `COMP_TYPE` for component types:
- `P` for point
- `G` for Gaussian
- `S` for shapelet (Jack only)

Jack and LoBES fits formats use the columns `MAJOR_DC`, `MINOR_DC` and `PA_DC`
for Gaussian component sizes and position angles (in degrees), while Gleam
fits files use `a`, `b` (arcseconds) and `pa` (degrees).

In an image space where RA increases from right to left (i.e. bigger RA
values are on the left), position angles rotate counter clockwise. A
position angle of 0 has the major axis aligned with the declination axis.

## Flux density models

Jack and LoBES fits formats use the column `MOD_TYPE` for flux density types:
- `pl` for power law
- `cpl` for curved power law
- `nan` for lists

Jack and LoBES fits formats use the columns `NORM_COMP_PL` and `ALPHA_PL` for
power law flux density normalisation and spectral index; and `NORM_COMP_CPL`,
`ALPHA_CPL` and `CURVE_CPL` for curved power law flux density normalisation,
while Gleam fits files use `S_200`, `alpha` and `beta`.

A reference frequency of 200MHz is assumed in all fits files.

Jack and LoBES fits formats use the columns `INT_FLXnnn` for integrated flux
densities in Jy at frequencies `nnn` MHz, while Gleam fits files use only `s_200`.
These columns are used to construct flux lists if power law information is
missing, or `MOD_TYPE` is `nan`.

Only Stokes I can be specified in fits sourcelists, Stokes Q, U and V are
assumed to have values of 0.

## Examples

Example Python code to display these files is in the [examples
directory](https://github.com/MWATelescope/mwa_hyperdrive/tree/main/examples).

e.g. `python examples/read_fits_srclist.py test_files/jack.fits`

| UNQ_SOURCE_ID   | NAME          |   RA |   DEC |   INT_FLX100 |   INT_FLX150 |   INT_FLX200 |   MAJOR_DC |   MINOR_DC |   PA_DC | MOD_TYPE   | COMP_TYPE   |   NORM_COMP_PL |   ALPHA_PL |   NORM_COMP_CPL |   ALPHA_CPL |   CURVE_CPL |
|-----------------|---------------|------|-------|--------------|--------------|--------------|------------|------------|---------|------------|-------------|----------------|------------|-----------------|-------------|-------------|
| point-list      | point-list_C0 |    0 |     1 |          3   |          2   |            1 |          0 |          0 |       0 | nan        | P           |              1 |        0   |               0 |         0   |         0   |
| point-pl        | point-pl_C0   |    1 |     2 |          3.5 |          2.5 |            2 |          0 |          0 |       0 | pl         | P           |              2 |       -0.8 |               0 |         0   |         0   |
| point-cpl       | point-cpl_C0  |    3 |     4 |          5.6 |          3.8 |            3 |          0 |          0 |       0 | cpl        | P           |              0 |        0   |               3 |        -0.9 |         0.2 |
| gauss-list      | gauss-list_C0 |    0 |     1 |          3   |          2   |            1 |         20 |         10 |      75 | nan        | G           |              1 |        0   |               0 |         0   |         0   |
| gauss-pl        | gauss-pl_C0   |    1 |     2 |          3.5 |          2.5 |            2 |         20 |         10 |      75 | pl         | G           |              2 |       -0.8 |               0 |         0   |         0   |
| gauss-cpl       | gauss-cpl_C0  |    3 |     4 |          5.6 |          3.8 |            3 |         20 |         10 |      75 | cpl        | G           |              0 |        0   |               3 |        -0.9 |         0.2 |
| shape-pl        | shape-pl_C0   |    1 |     2 |          3.5 |          2.5 |            2 |         20 |         10 |      75 | pl         | S           |              2 |       -0.8 |               0 |         0   |         0   |
| shape-pl        | shape-pl_C1   |    1 |     2 |          3.5 |          2.5 |            2 |         20 |         10 |      75 | pl         | S           |              2 |       -0.8 |               0 |         0   |         0   |

| NAME        |   N1 |   N2 |   COEFF |
|-------------|------|------|---------|
| shape-pl_C0 |    0 |    0 |     0.9 |
| shape-pl_C0 |    0 |    1 |     0.2 |
| shape-pl_C0 |    1 |    0 |    -0.2 |
| shape-pl_C1 |    0 |    0 |     0.8 |

e.g. `python examples/read_fits_srclist.py test_files/gleam.fits`

| Name      |   RAJ2000 |   DEJ2000 |   S_200 |   alpha |   beta |     a |     b |   pa |
|-----------|-----------|-----------|---------|---------|--------|-------|-------|------|
| point-pl  |         1 |         2 |       2 |    -0.8 |    0   |     0 |     0 |    0 |
| point-cpl |         3 |         4 |       3 |    -0.9 |    0.2 |     0 |     0 |    0 |
| gauss-pl  |         1 |         2 |       2 |    -0.8 |    0   | 72000 | 36000 |   75 |
| gauss-cpl |         3 |         4 |       3 |    -0.9 |    0.2 | 72000 | 36000 |   75 |

these are both equivalent to the following YAML file (ignoring shapelets and
lists for the gleam example):

```yaml
point-list:
- ra: 0.0
  dec: 1.0
  comp_type: point
  flux_type:
    list:
    - freq: 100000000.0
      i: 3.0
    - freq: 150000000.0
      i: 2.0
    - freq: 200000000.0
      i: 1.0
point-pl:
- ra: 1.0
  dec: 2.0
  comp_type: point
  flux_type:
    power_law:
      si: -0.8
      fd:
        freq: 200000000.0
        i: 2.0
point-cpl:
- ra: 3.0000000000000004
  dec: 4.0
  comp_type: point
  flux_type:
    curved_power_law:
      si: -0.9
      fd:
        freq: 200000000.0
        i: 3.0
      q: 0.2
gauss-list:
- ra: 0.0
  dec: 1.0
  comp_type:
    gaussian:
      maj: 72000.0
      min: 36000.0
      pa: 75.0
  flux_type:
    list:
    - freq: 100000000.0
      i: 3.0
    - freq: 150000000.0
      i: 2.0
    - freq: 200000000.0
      i: 1.0
gauss-pl:
- ra: 1.0
  dec: 2.0
  comp_type:
    gaussian:
      maj: 72000.0
      min: 36000.0
      pa: 75.0
  flux_type:
    power_law:
      si: -0.8
      fd:
        freq: 200000000.0
        i: 2.0
gauss-cpl:
- ra: 3.0000000000000004
  dec: 4.0
  comp_type:
    gaussian:
      maj: 72000.0
      min: 36000.0
      pa: 75.0
  flux_type:
    curved_power_law:
      si: -0.9
      fd:
        freq: 200000000.0
        i: 3.0
      q: 0.2
shape-pl:
- ra: 1.0
  dec: 2.0
  comp_type:
    shapelet:
      maj: 72000.0
      min: 36000.0
      pa: 75.0
      coeffs:
      - n1: 0
        n2: 0
        value: 0.9
      - n1: 0
        n2: 1
        value: 0.2
      - n1: 1
        n2: 0
        value: -0.2
  flux_type:
    power_law:
      si: -0.8
      fd:
        freq: 200000000.0
        i: 2.0
- ra: 1.0
  dec: 2.0
  comp_type:
    shapelet:
      maj: 72000.0
      min: 36000.0
      pa: 75.0
      coeffs:
      - n1: 0
        n2: 0
        value: 0.8
  flux_type:
    power_law:
      si: -0.8
      fd:
        freq: 200000000.0
        i: 2.0
```