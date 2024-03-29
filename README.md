### Abstract ###
The goal of automated emotion recognition research is to endow machines with emotional intelligence, allowing them to understand human emotions. Recognizing emotions can be approached using signals of external manifestations, such as speech and facial expressions, or internal manifestations, by analyzing EEG. Most of the affective computing models are based on images, audio, videos and brain signals, collected in a laboratory environment. Nowadays, advancements in mobile computing and wearable technologies have enabled the continuous monitoring of physiological signals. There is an emerging necessity of affective computing models that focus on utilizing only peripheral physiological signals for emotion recognition. In this study, an emotion classification method based on neural networks and exploiting peripheral physiological signals, obtained by wearable devices, is presented. An Attention-based Long Short-Term Memory classification architecture is proposed to accurately predict emotions in real-time into binary levels of the 2 dimensional arousal-valence space. K-EmoCon dataset was used throughout the project and the data were gathered from 16 sessions of approximately 10-minute long paired debates between 2 participants, on a social issue. The dataset includes emotion annotations from all three available perspectives: self, debate partner, external observers and their impact on classification performance was analyzed. Results show that our method has a measured accuracy of ~91% for binary classification, outperforming a traditional machine learning approach and a simplified Long Short-Term Memory classification architecture on the same task.

#### Supplementary Codes for the K-EmoCon Dataset Descriptor
- [chauvenet.py](https://github.com/sotirismos/Emotion-Recognition-Conversations/blob/master/K-EmoCon_SupplementaryCodes/utils/chauvenet.py) - an implementation of [Chauvenet's criterion](https://en.wikipedia.org/wiki/Chauvenet%27s_criterion) for detecting outliers.
- [vote_majority.py](https://github.com/sotirismos/Emotion-Recognition-Conversations/blob/master/K-EmoCon_SupplementaryCodes/utils/vote_majority.py) - implements a majority voting to get a consensus between external annotations.
- [plotting.py](https://github.com/sotirismos/Emotion-Recognition-Conversations/blob/master/K-EmoCon_SupplementaryCodes/utils/plotting.py) - includes functions to produce the IRR heatmap (Fig. 4) in the K-EmoCon dataset descriptor.
    - `get_annotations` - loads annotations saved as CSV files.
    - `subtract_mode_from_values` - implements mode subtraction.
    - `compute_krippendorff_alpha` - computes Krippendorff's alpha (IRR).
    - `plot_heatmaps` - plots the IRR heatmap.

#### Preprocessing
Running [preprocess.py](https://github.com/sotirismos/Emotion-Recognition-Conversations/blob/master/preprocess.py) will will create 5-second segments containing 4 types of biosignals `['bvp', 'eda', 'temp', 'ecg']` acquired during debates as JSON files under subdirectories corresponding to each participant.

JSON files for biosignal segments will have names with the following pattern: for example `p01-017-24333.json` indicates that the file is a 17th 5-second biosignal segment for participant 1.

The last 6 digits are multiperspective emotion annotations associated with the segment, in the order of 1) self-arousal, 2) self-valence, 3) partner-arousal, 4) partner-valence, 5) external-arousal, and 6) external-valence.

#### Classification results

**SVM**

| Affect | Annotation perspective    | Accuracy    | F1-Score    | AUROC    |
| :---:   | :---: | :---: | :---: | :---: |
| Arousal | Self   | 0.679   | 0.6059    | 0.6257    |
| Arousal | Partner   | 0.6953   | 0.619    | 0.65    |
| Valence | Self   | 0.7565   | 0.70999    | 0.675    |
| Valence | Partner   | 0.76   | 0.7105    | 0.683    |

**XGBoost**

| Affect | Annotation perspective    | Accuracy    | F1-Score    | AUROC    |
| :---:   | :---: | :---: | :---: | :---: |
| Arousal | Self   | 0.786   | 0.772    | 0.8672    |
| Arousal | Partner   | 0.812   | 0.7984    | 0.879    |
| Valence | Self   | 0.81   | 0.814    | 0.8055    |
| Valence | Partner   | 0.805   | 0.8103    | 0.8164    |

**LSTM**

| Affect | Annotation perspective    | Accuracy    | F1-Score    | AUROC    |
| :---:   | :---: | :---: | :---: | :---: |
| Arousal | Self   | 0.822   | 0.8235    | 0.882    |
| Arousal | Partner   | 0.8402   | 0.834    | 0.8973    |
| Valence | Self   | 0.8351   | 0.8202    | 0.8152    |
| Valence | Partner   | 0.8302   | 0.8258    | 0.8243    |

**Attention-based LSTM**

| Affect | Annotation perspective    | Accuracy    | F1-Score    | AUROC    |
| :---:   | :---: | :---: | :---: | :---: |
| Arousal | Self   | 0.8777   | 0.915    | 0.905    |
| Arousal | Partner   | 0.903   | 0.9323    | 0.9241    |
| Valence | Self   | 0.9185   | 0.9165    | 0.9423    |
| Valence | Partner   | 0.9154   | 0.9133    | 0.9391    |

#### Thesis full report (Greek) is available [online](https://ikee.lib.auth.gr/record/340502/?ln=en)


