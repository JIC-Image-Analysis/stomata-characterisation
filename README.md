# Stomata finding

This code is intended to locate stomata in confocal image stacks.

## Discussion

We started out by identifying the stomata outlines. For example in series 8 we
know that region 9 represent stomata number 1.

```
python scripts/find_stomata.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 8 9
```

We can then look at the bright field profile lines across all z-slices for a
particular stomata at a particular time point. For example stomata 1 at its
first time point.

```
python scripts/plot_profile_lines.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 0 0
```

We can then calculate the opening from the average bright field peak.
For example for stomata 1 at its first time point.

```
python scripts/calculate_opening.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 0 -t 0
```

Or for all time points in a particular stomata. For example stomata 1.

```
python scripts/calculate_opening.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 0
```

Finally to view the calculated opening across all z-slices for a particular
stomata we can generate augmented images.

```
python scripts/generate_augmented_zstack_images.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 0 0 -d tmp
```

The images can be view using a viewer such as ``eye/viewer``:

https://github.com/mrmh2/eye/blob/master/lib/viewer.py


## Testing

To test that things are working as expected. Put the
``2014-03-20-Lti6b-GFP-ABA-time-series.lif`` file in the ``data/raw``
directory. Then run the tests using ``nosetests``.

```
nosetests
```
