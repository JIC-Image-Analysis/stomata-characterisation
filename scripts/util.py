import os
import errno

STOMATA = (
    dict(region=9, series=range(8, 15)),     # Stomata id 1
    dict(region=14, series=range(8, 15)),    # Stomata id 2
    dict(region=20, series=range(8, 15)),    # Stomata id 3
    dict(region=8, series=range(15, 24)),    # Stomata id 4
    dict(region=17, series=range(15, 24)),   # Stomata id 5
    dict(region=21, series=range(15, 24)),   # Stomata id 6
    None,                                    # Unable to identify stomata 7
    dict(region=8, series=range(24, 36)),    # Stomata id 8
)

def stomata_lookup(stomata_id):
    """Return stomata region and series information."""
    d = STOMATA[stomata_id]
    if d is None:
        raise(NotImplementedError("Have not found this stomata yet"))
    return d["region"], d["series"]


def safe_mkdir(dir_path):

    try:
        os.makedirs(dir_path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            print "Error creating directory %s" % dir_path
            sys.exit(2)
