# Color quantization

This is a study & experiment about various formulas for optimizing median cut
algorithm heuristics. In particular, this is an experiment on which logic is
the most appropriate for next box selection and axis cut selection.


## Tree

**Note**: the scripts depend on Python3, `matplotlib` and `PIL`

- `pal.py`: core lib where all the important code is
- `gen_results.py`: generate results, see `--help` for more info
- `show_results.py`: show or export the results in an image, see `--help` for
  more info
- `show_pal.py`: tool to display a palette stored in a 16×16 file in sRGB,
  Linear and OkLab
- `samples`: some sample files


## Algorithms tested

Each algo follows the `<axis-selection>_<operator>_<measurement>` naming, where:

- `axis-selection`:
    + `xer2`: cut the axis with the maximum error squared
    + `xlen`: cut the axis with the maximum length
- `operator`:
    + `max`: maximum measurement out of all the channels
    + `mul`: product of the measurements of all the channels
    + `sum`: sum of the measurements of all the channels
- `measurement`:
    + `er2absW`: error squared, honoring weights
    + `er2noW`: error squared, not honoring weights
    + `er2normW`: error squared, honoring weights, normalized
    + `len`: length of the axis


## Results

![Results](img/results.png)


## Conclusions

1. Overall, non-normalized weighted sum of the squared error is the best box
   selection algorithm
2. Overall, cutting the axis with the biggest weighted sum of squared error is
   the best axis cut selection algorithm
