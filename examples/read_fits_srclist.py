from astropy.io import fits
import sys
from tabulate import tabulate

for hdu in fits.open(sys.argv[-1])[1:]:
    print(tabulate(hdu.data, headers=[c.name for c in hdu.columns], tablefmt="github"))