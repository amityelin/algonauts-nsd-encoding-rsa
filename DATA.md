# Data access and derived outputs

## Source data

This project uses the Algonauts 2023 challenge data, derived from the Natural Scenes Dataset
(NSD). Start with the official [Algonauts Project 2023 site](https://algonauts.csail.mit.edu/)
for challenge context and the official [NSD access page](https://naturalscenesdataset.org/)
for the current dataset route. Complete the NSD Data Access Agreement, follow the access
instructions supplied by the dataset maintainers, and read the
[NSD Data Manual](https://naturalscenesdataset.org/NSD_Data_Manual_v1.5.pdf) before downloading.
Then place only the required Algonauts development-kit/tutorial files in the layout below.
Review the applicable NSD, Algonauts, COCO-image, and institutional terms before use.

The repository does not redistribute source images, fMRI matrices, ROI masks, pretrained
weights, or participant-level raw data.

Expected local layout:

```text
algonauts_2023_tutorial_data/
├── subj01/
│   ├── training_split/
│   │   ├── training_fmri/
│   │   └── training_images/
│   └── roi_masks/
└── ... subj02 through subj08
```

Do not copy the downloaded data into this repository. Set `ALGONAUTS_DATA_ROOT` to the
directory above and `ALGONAUTS_OUTPUT_ROOT` to a separate writable directory for derived
artifacts. Alternatively, pass both roots explicitly to `get_paths`. Colab users may mount
Drive; the original Drive-shortcut defaults are used only after Colab has actually loaded
`google.colab`. Local runs never infer paths from the current directory or `/content`.

Resolution order for each root is: explicit argument, corresponding environment variable,
then the Colab-only default. If none applies, configuration raises an error explaining how to
set the missing root.

## Derived outputs

Large derived outputs remain external because feature arrays, response matrices, and RDMs are
not suitable for normal Git storage. The existing shared folder is linked from
`algonauts outputs/link _to_folder.txt`.

Before redistributing any derived artifact, verify that it does not reproduce restricted source
data and that the relevant dataset and model-weight terms permit sharing. The small CSV and JSON
files under `results/` contain only aggregate preliminary findings and run metadata.

## Privacy

Subject identifiers are the public dataset pseudonyms (`subj01`–`subj08`). Do not add direct
identifiers, local credentials, access tokens, private Drive metadata, or raw participant data to
the repository.
