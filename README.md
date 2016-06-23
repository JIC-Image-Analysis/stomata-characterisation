# Stomata characterisation

This code is intended to determine how open a stoma is from a stack of
brightfield images.

## Installation

Install ``freeimage``.

Install the Python dependencies.

```
numpy==1.9.2
scipy==0.15.1
matplotlib==1.4.3
scikit-image==0.11.3
cv2==2.4.7
jicimagelib=0.3.1
```

Clone this repository.

```
git clone git@githq.nbi.ac.uk:rg-matthew-hartley/find-stomata.git
```

## Run analysis

Create a data directory structure.

```
cd find-stomata
mkdir -p data/raw
```

Place the ``2014-03-20-Lti6b-GFP-ABA-time-series.lif`` file the directory
``data/raw``.

Run the ``analyse_all_stomata.py`` script.

```
python scripts/analyse_all_stomata.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif out
```

This produces the output directory ``out``.

Inspect the images that show the stomata opening analysis performed.

```
out/opening_stomate_1_timepoint_0.png
...
```

Note that the algorithm does not find the correct minima for 

```
out/opening_stomate_8_timepoint_0.png
```

Re-run the analysis for this stomate and time point with a higher Gaussian
sigma smoothing for the average profile line.

```
python scripts/calculate_opening.py data/raw/2014-03-20-Lti6b-GFP-ABA-time-series.lif 7 0 -s 2
```

Note that the stomate identifier is zero-indexed in the command above, i.e. 7
represents stomate 8.

Update the opening value for stomate 8 time point zero in the csv file
``out/stomata_openings.csv``.

## Configuration

Configurations such as the stomate center, scale factor, z-slices to be
included, etc, are located in the global variable ``util.STOMATA`` in the file
``scripts/util/__init__.py``.
