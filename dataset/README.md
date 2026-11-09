# POLAR-Synth

POLAR is trained on **POLAR-Synth**, a synthetic dataset of reference/candidate radiology report pairs with LLM-annotated error counts. It is built from four public report datasets:

| Subset | Modality | Source | Availability |
|---|---|---|---|
| `ct-rate` | Chest CT | [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | 🤗 [lbuess/POLAR-Synth](https://huggingface.co/datasets/lbuess/POLAR-Synth) |
| `mr-rate` | Brain / Spine MRI | [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE) | 🤗 [lbuess/POLAR-Synth](https://huggingface.co/datasets/lbuess/POLAR-Synth) |
| `cxr` | Chest X-ray | [GREEN](https://huggingface.co/datasets/StanfordAIMI/GREEN) (reports from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)) | generation code only |
| `merlin` | Abdominal CT | [MERLIN](https://github.com/StanfordMIMI/Merlin/blob/main/documentation/download.md) | generation code only |

The `cxr` subset uses the GREEN report pairs, which are taken from MIMIC-CXR. The licenses of MIMIC-CXR and MERLIN do not allow redistributing their reports, so these two subsets are not published. If you have access to the source data, you can rebuild them with the generation code below.

## License

The released subsets (`ct-rate`, `mr-rate`) inherit the license of their source data: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Rebuilt `cxr` and `merlin` subsets are subject to the data use agreements of MIMIC-CXR and MERLIN.

## Download

The dataset is gated: first accept the terms on the [dataset page](https://huggingface.co/datasets/lbuess/POLAR-Synth) and log in with `hf auth login`. Then download the public subsets (`ct-rate`, `mr-rate`) into the training layout, from the repository root:

```bash
python dataset/download.py                          # train and val splits
python dataset/download.py --splits train val test  # including the test split
```

## Training data format

[POLAR training](../polar/README.md#training) reads CSV files from this folder:

```
dataset/train/{cxr,ct-rate,mr-rate,merlin}_train.csv
dataset/val/{cxr,ct-rate,mr-rate,merlin}_val.csv
```

| Column | Content |
|---|---|
| `reference`, `candidate` | report texts |
| `conditioning` | `full report` or an anatomy label |
| `sig_a` … `sig_h` | error counts per category |
| `matched_normal`, `matched_abnormal` | matched-finding counts |

File paths and column names are set in [polar/train/conf/data/default.yaml](../polar/train/conf/data/default.yaml). To train on fewer subsets, remove the missing files from `train_files` and `val_files` there.

## Dataset generation

<!-- TODO: add the dataset generation pipeline and describe it here -->
The code used to generate POLAR-Synth, including the `cxr` and `merlin` subsets, will be added to this folder.
