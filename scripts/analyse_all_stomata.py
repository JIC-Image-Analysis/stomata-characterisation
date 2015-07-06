"""Calculate opening of all stomata."""

import argparse
import os
import os.path

import matplotlib.pyplot as plt

from jicimagelib.io import AutoWrite, AutoName

from util import (
    STOMATA,
    unpack_data,
    stomata_timeseries_lookup,
)

from calculate_opening import StomateOpening

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("confocal_file", help="File containing confocal data")
    parser.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir
    AutoWrite.on = False

    image_collection = unpack_data(args.confocal_file)

    csv_fpath = os.path.join(AutoName.directory, "stomata_openings.csv")
    with open(csv_fpath, "w") as fh:
        fh.write("StomateId,Timepoint,Opening(uM)\n")
        for i in range(len(STOMATA)):
            stomata_timeseries = stomata_timeseries_lookup(i)
            for timepoint, s in enumerate(stomata_timeseries.series):

                stomate_opening = StomateOpening(image_collection, i,
                    timepoint, sigma=1.0)

                # Write csv file.
                csv_line = "{},{},{:.3f}\n".format(
                    stomate_opening.stomate.stomate_id,
                    stomate_opening.stomate.timepoint_id,
                    stomate_opening.opening_distance)
                fh.write(csv_line)

                # Save figure.
                stomate_opening.plot()
                fname = "opening_stomate_{}_timepoint_{}.png".format(
                    i+1, timepoint)
                fpath = os.path.join(AutoName.directory, fname)
                plt.savefig(fpath)
                plt.close()

if __name__ == "__main__":
    main()
