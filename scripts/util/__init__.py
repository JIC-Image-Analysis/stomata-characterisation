"""Utility functions for the stomata scritps."""

import os
import errno

from jicimagelib.image import DataManager
from jicimagelib.io import FileBackend

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', '..', 'data', 'unpack')

STOMATA = (
    dict(center=(119, 141),
        series=range(8, 15),
        scale_factor=0.1571,
        zslice_include=range(13, 20),
    ),  # Stomata id 1
    dict(center=(420, 257),
        series=range(8, 15),
        scale_factor=0.1571,
        zslice_include=range(8, 17),
    ),  # Stomata id 2
    dict(center=(219, 426),
        series=range(8, 15),
        scale_factor=0.1571,
        zslice_include=range(8, 17),
    ),  # Stomata id 3
    dict(center=(329, 127),
        series=range(15, 24),
        scale_factor=0.1789,
        zslice_include=range(5, 13),
    ),  # Stomata id 4
    dict(center=(396, 325),
        series=range(15, 24),
        scale_factor=0.1789,
        zslice_include=range(5, 15),
    ),  # Stomata id 5
    dict(center=(89, 376),
        series=range(15, 24),
        scale_factor=0.1789,
        zslice_include=range(5, 13),
    ),  # Stomata id 6
    dict(center=(153, 120),
        series=range(24, 36),
        scale_factor=0.1561,
        zslice_include=range(10, 15),
    ),  # Stomata id 7
    dict(center=(305, 349),
        series=range(24, 36),
        scale_factor=0.1561,
        zslice_include=range(5, 15),
    ),  # Stomata id 8
)

class BaseStomate(object):
    """Base stomate class."""
    def __init__(self, identifier, center, series, scale_factor, zslice_include):
        self.identifier = identifier
        self.center = center
        self.series = series
        self.scale_factor = scale_factor
        self.zslice_include = zslice_include

    @property
    def stomate_id(self):
        """Return the stomate identifier."""
        return self.identifier

    @property
    def x(self):
        """Return x coordinate of center."""
        return self.center[0]
        
    @property
    def y(self):
        """Return y coordinate of center."""
        return self.center[1]
        

class StomateTimePoint(BaseStomate):
    """Stomate at a particular time point."""
    @property
    def stomate_id(self):
        """Return the stomate identifier."""
        return self.identifier[0]

    @property
    def timepoint_id(self):
        """Return the timepoint identifier."""
        return self.identifier[1]

    def __str__(self):
        return "<StomateTimePoint(stomate={}, timepoint={})>".format(
            self.stomate_id, self.timepoint_id)

class StomataTimeSeries(BaseStomate):
    """Stomata time series."""
    def stomate_at_timepoint(self, timepoint):
        """Return stomate at timepoint id."""
        for i, series in enumerate(self.series):
            if timepoint == i:
                return StomateTimePoint(
                    identifier=(self.stomate_id, timepoint),
                    center=self.center,
                    series=series,
                    scale_factor=self.scale_factor,
                    zslice_include=self.zslice_include)
        raise(IndexError("No such timepoint: {}".format(timepoint)))
        
    def __iter__(self):
        for t in range(len(self.series)):
            yield self.stomate_at_timepoint(t)

    def __str__(self):
        return "<StomataTimeSeries(stomata={}, timepoints={})>".format(
            self.stomate_id, range(len(self.series)))

def stomata_timeseries_lookup(stomata_id):
    """Return stomata time series."""
    stomata = STOMATA[stomata_id]
    return StomataTimeSeries(
        stomata_id+1,
        stomata["center"],
        stomata["series"],
        stomata["scale_factor"],
        stomata["zslice_include"])

def stomate_lookup(stomata_id, timepoint_id):
    """Lookup a particular stomate at a particular timepoint."""
    stomate_timeseries = stomata_timeseries_lookup(stomata_id)
    return stomate_timeseries.stomate_at_timepoint(timepoint_id)

    

def safe_mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            print "Error creating directory %s" % dir_path
            sys.exit(2)

def unpack_data(confocal_file):
    """Unpack the file and return an image collection object."""
    safe_mkdir(UNPACK)
    backend = FileBackend(UNPACK)
    data_manager = DataManager(backend)
    data_manager.load(confocal_file)
    image_collection = data_manager[0]

    return image_collection

