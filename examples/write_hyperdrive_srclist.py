#!/usr/bin/env python

# from the pyyaml package
import yaml

# You can write the source list out manually.
source_list = {
    "super_sweet_source1": [{
        "ra": 10.0,
        "dec": -27.0,
        "comp_type": "point",
        "flux_type": {
            "list": [{
                "freq": 180e6,
                "i": 10.0,
                "q": 0.0,
                "u": 0.0,
                "v": 0.0,
            }]
        }
    }]
}

# You can add sources to the source list like so.
second_source = [{
    "ra": 0,
    "dec": -35.0,
    "comp_type": {
        "gaussian": {
            "maj": 20,
            "min": 10,
            "pa": 75,
        }
    },
    "flux_type": {
        "list": [{
            "freq": 150e6,
            "i": 1,
            "q": 2,
            "u": 3,
            "v": 4,
        }]
    }
}]
source_list["super_sweet_source2"] = second_source


# At this point, some functions are helpful.
def make_flux_density(freq, i, q, u, v):
    return {
        "freq": freq,
        "i": i,
        "q": q,
        "u": u,
        "v": v,
    }


def make_power_law(si, freq, i, q, u, v):
    return {
        "si": si,
        "fd": make_flux_density(freq, i, q, u, v)
    }


# Add another flux density to the list of the first source's first component.
source_list["super_sweet_source1"][0]["flux_type"]["list"].append(make_flux_density(170e6, 5, 1, 2, 3))

# Replace the flux type of the second source's first component with a power law.
del source_list["super_sweet_source2"][0]["flux_type"]["list"]
source_list["super_sweet_source2"][0]["flux_type"]["power_law"] = make_power_law(-0.8, 170e6, 5, 1, 2, 3)

# Add another component with a shapelet and curved power law.
third_comp = {
    "ra": 155,
    "dec": -10.0,
    "comp_type": {
        "shapelet": {
            "maj": 20,
            "min": 10,
            "pa": 75,
            "coeffs": [
                {
                    "n1": 0,
                    "n2": 1,
                    "value": 0.5,
                }
            ]
        }
    },
    "flux_type": {
        "curved_power_law": {
            "si": -0.6,
            "fd": make_flux_density(150e6, 50, 0.5, 0.1, 0),
            "q": 0.2,
        }
    }
}
source_list["super_sweet_source2"].append(third_comp)

# What does the yaml file look like?
print(yaml.dump(source_list))

# The following writes the yaml file to the specified path.
with open("/tmp/hyperdrive_srclist.yaml", "w") as h:
    yaml.dump(source_list, stream=h)

# All of the above code works when you change yaml for json.
import json
with open("/tmp/hyperdrive_srclist.json", "w") as h:
    # The following line works, but the output is hard to read.
    # json.dump(source_list, fp=h)
    json.dump(source_list, fp=h, indent=4)
