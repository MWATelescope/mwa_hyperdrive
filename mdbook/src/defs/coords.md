# Coordinate systems

## Antenna/tile/station/array coordinates

In measurement sets and uvfits files, antennas/tiles/stations usually have their
positions recorded in the
[ITRF](https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame)
frame (internally we refer to this as "geocentric XYZ"). There's also a
"geodetic XYZ" frame; an example of this is [WGS
84](https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84) (which we assume
everywhere when converting, as it's the current best ellipsoid). Finally,
there's also an "East North Height" coordinate system.

To calculate UVW baseline coordinates, geodetic XYZ coordinates are
required[^1]. Therefore, various coordinate conversions are required to obtain
UVWs. The conversion between all of these systems is briefly described below.
The relevant code lives within [`Marlu`](https://github.com/MWATelescope/Marlu).

## ITRF and "geocentric XYZ"

As the name implies, this coordinate system uses the centre of the Earth as a
reference. To convert between geocentric and geodetic, an array position is
required (i.e. the "average" location on the Earth of the instrument collecting
visibilities). When all antenna positions are geocentric, the array position is
given by the mean antenna position.

Measurement sets indicate the usage of ITRF with the "MEASURE_REFERENCE" keyword
attached to the POSITION column of an ANTENNA table (value "ITRF").

The [uvfits
standard](https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf) states
that only supported frame is "ITRF", and `hyperdrive` assumes that only ITRF is
used. However, CASA/casacore seem to write out antenna positions incorrectly;
the positions look like what you would find in an equivalent measurement set.
The incorrect behaviour is detected and accounted for.

## "Geodetic XYZ"

This coordinate system is similar to geocentric, but uses an array position as
its reference.

Measurement sets support the WGS 84 frame, again with the "MEASURE_REFERENCE"
keyword attached to the POSITION column of an ANTENNA table (value "WGS84").
However, `hyperdrive` currently does not check if geodetic positions are used;
it instead just assumes geocentric.

When read literally, the antenna positions in a uvfits file ("STABXYZ" column of
the "AIPS AN" HDU) *should* be geodetic, not counting the aforementioned
casacore bug.

## East North Height (ENH)

MWA tiles positions are listed in [metafits](../defs/mwa/metafits.md) files with
the ENH coordinate system. Currently, ENH coordinates are converted to geodetic
XYZ with the following pseudocode:

```python
x = -n * sin(latitude) + h * cos(latitude)
y = e
z = n * cos(latitude) + h * sin(latitude)
```

(I unfortunately don't know anything more about this system.)

## Array positions

Array positions can be found with the mean geocentric antenna positions, as is
the case with measurement sets, or with the `ARRAYX`, `ARRAYY` and `ARRAYZ` keys
in a uvfits file. However, `hyperdrive` allows the user to supply a custom array
position, which will be used in any conversions between provided antenna
positions and other coordinate systems as required.

For raw MWA data, no array position is supplied, so we assume a location for the
MWA. This is currently:

- Latitude: -0.4660608448386394 radians (or −26.70331941 degrees)
- Longitude: 2.0362898668561042 radians (or 116.6708152 degrees)
- Height: 377.827 metres

## Precession

It is often necessary to precess antenna positions to the J2000 epoch, because:

- Measurement sets and uvfits expect their UVWs to be specified in the J2000 epoch; and
- Sky model source lists are expected to be specified in the J2000 epoch.

`hyperdrive` performs precession on each timestep of input visibility data to
(hopefully) get UVWs as correct as possible.

The process to precess geodetic XYZs is too complicated to detail here, but the
code lives within [`Marlu`](https://github.com/MWATelescope/Marlu). This code is
a re-write of old MWA code, and there appears to be no references on how or why
it works; any information is greatly appreciated!

## UVWs

A geodetic XYZ is converted to UVW using the following pseudocode:

```python
s_ha = sin(phase_centre.hour_angle)
c_ha = cos(phase_centre.hour_angle)
s_dec = sin(phase_centre.declination)
c_dec = cos(phase_centre.declination)

u = s_ha * x + c_ha * y,
v = -s_dec * c_ha * x + s_dec * s_ha * y + c_dec * z,
w = c_dec * c_ha * x - c_dec * s_ha * y + s_dec * z,
```

Note that this is a UVW coordinate for an antenna. To get the proper baseline
UVW, a difference between two antennas' UVWs needs to be taken. The order of
this subtraction is important; `hyperdrive` uses the "antenna1 - antenna2"
convention. Software that reads data may need to conjugate visibilities if this
convention is different.

### Further reading

- <https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame>
- <https://en.wikipedia.org/wiki/World_Geodetic_System>
- The [uvfits
  standard](https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf)
- <https://casa.nrao.edu/Memos/CoordConvention.pdf>
- <https://casa.nrao.edu/Memos/229.html#SECTION00042000000000000000>

[^1]: If this isn't true, _please_ file a `hyperdrive` issue.
