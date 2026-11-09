import torch
from torch.utils.data import Dataset
import pandas as pd

# Custom imports
from polar.train.dataset_utils import sample_anatomy_text


class PolarDataset(Dataset):
    def __init__(
        self,
        csv_files,
        tokenizer,
        ref_col,
        cand_col,
        error_cols,
        region_col=None,
        max_length=512,
    ):
        """
        Args:
            csv_files (List[str]): List of paths to CSV files with data.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            ref_col (str): Column name for the reference report.
            cand_col (str): Column name for the candidate report.
            error_cols (list of str): List of column names for the error counts.
            region_col (str, optional): Column name for the region conditioning.
            max_length (int): Max length for tokenization.
        """

        self.tokenizer = tokenizer
        self.ref_col = ref_col
        self.cand_col = cand_col
        self.error_cols = list(error_cols)
        self.region_col = region_col
        self.max_length = max_length

        # Read all CSV files into a single DataFrame
        self.data = pd.concat(
            [pd.read_csv(file) for file in csv_files], ignore_index=True
        )

        # CSV column validation
        required_cols = [self.ref_col, self.cand_col, *self.error_cols]
        if self.region_col:
            required_cols.append(self.region_col)
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(
                "Missing required columns in CSV. "
                f"Missing: {missing}. "
                f"Available: {list(self.data.columns)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get reference and candidate text
        reference_text = f"Reference: {str(row[self.ref_col])}"
        candidate_text = f"Candidate: {str(row[self.cand_col])}"

        # Get condition text (support both "full report" and "full_report" labels)
        raw_region = str(row[self.region_col]).strip()
        normalized_region = raw_region.lower().replace("_", " ")
        if normalized_region == "full report":
            condition_text = "Condition: full report"
        else:
            anatomy_text = sample_anatomy_text(raw_region)
            condition_text = f"Condition: {anatomy_text}"

        # If tokenizer has a sep_token, use it to join the text, otherwise use \n
        if hasattr(self.tokenizer, "sep_token"):
            sep = f" {self.tokenizer.sep_token} "
            joint_text = sep.join([condition_text, reference_text, candidate_text])
        else:
            joint_text = f"{condition_text}\n{reference_text}\n{candidate_text}\n"

        # Tokenize input
        encoding = self.tokenizer(
            joint_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Get labels (multi-head regression targets)
        labels = row[self.error_cols]

        # Robust conversion (handles strings/NaNs)
        labels = pd.to_numeric(labels, errors="coerce").fillna(0.0).astype("float32")
        labels = torch.tensor(labels.values, dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
