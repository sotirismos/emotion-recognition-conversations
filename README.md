## Supplementary Codes for the K-EmoCon Dataset Descriptor
- [chauvenet.py](https://github.com/sotirismos/Emotion-Recognition-Conversations/blob/master/K-EmoCon_SupplementaryCodes/utils/chauvenet.py) - an implementation of [Chauvenet's criterion](https://en.wikipedia.org/wiki/Chauvenet%27s_criterion) for detecting outliers.
- vote_majority.py - implements a majority voting to get a consensus between external annotations.
- plotting.py - includes functions to produce the IRR heatmap (Fig. 4) in the K-EmoCon dataset descriptor.
get_annotations - loads annotations saved as CSV files.
subtract_mode_from_values - implements mode subtraction.
compute_krippendorff_alpha - computes Krippendorff's alpha (IRR).
plot_heatmaps - plots the IRR heatmap.
