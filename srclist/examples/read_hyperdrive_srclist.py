#!/usr/bin/env python

import sys
import json

# from the pyyaml package
import yaml

input_file = sys.argv[1]
file_extension = sys.argv[1].split(".")[-1]
if file_extension == "yaml":
    with open(input_file, "r") as h:
        # The following line works, but the yaml library complains.
        # source_list = yaml.load(h)
        source_list = yaml.load(h, Loader=yaml.SafeLoader)
elif file_extension == "json":
    with open(input_file, "r") as h:
        source_list = json.load(h)
else:
    print("Unrecognised file extension: {}".format(file_extension))
    sys.exit(1)

print("Number of sources in the source list: {}".format(len(source_list)))

print("")
print("Iterating over sources")
print("")
for source_name in source_list:
    print("Source name: {}".format(source_name))
    for i, component in enumerate(source_list[source_name]):
        print("Component {} (RA {}, Dec {})".format(i, component["ra"], component["dec"]))

        # Handle component type.
        if "point" in component["comp_type"]:
            # Do stuff with point components here.
            print("is a point source")

        elif "gaussian" in component["comp_type"]:
            g = component["comp_type"]["gaussian"]
            # Do stuff with gaussian components here.
            print("is a gaussian source: major {} [arcsec], minor {} [arcsec], position angle {} [degrees]"
                  .format(g["maj"], g["min"], g["pa"]))

        elif "shapelet" in component["comp_type"]:
            s = component["comp_type"]["shapelet"]
            # Do stuff with shapelet components here.
            print("is a shapelet source: major {} [arcsec], minor {} [arcsec], position angle {} [degrees], coefficients {}"
                  .format(s["maj"], s["min"], s["pa"], s["coeffs"]))

        else:
            print("Did not recognise component type: {}".format(component["comp_type"]))
            sys.exit(1)

        # Handle flux type.
        if "list" in component["flux_type"]:
            # Do stuff with the frequencies here.
            for f in component["flux_type"]["list"]:
                print("for freq {} [Hz], Stokes I {} [Jy], Q {}, U {}, V {}".format(*f.values()))

        elif "power_law" in component["flux_type"]:
            pl = component["flux_type"]["power_law"]
            # Do stuff with the power law here.
            print("with spectral index {} and at freq {} [Hz] Stokes I {} [Jy], Q {}, U {}, V {}"
                  .format(pl["si"], *pl["fd"].values()))

        elif "curved_power_law" in component["flux_type"]:
            cpl = component["flux_type"]["curved_power_law"]
            # Do stuff with the curved power law here.
            print("with spectral index {}, curvature paramater {},\nand at freq {} [Hz] Stokes I {} [Jy], Q {}, U {}, V {}"
                  .format(cpl["si"], cpl["q"], *cpl["fd"].values()))

        else:
            print("Did not recognise flux density type: {}".format(component["flux_type"]))
            sys.exit(1)

        print("")
