import random

_SEEN_UNKNOWN_ANATOMY = set()

ANATOMY_SYNONYMS = {
    "Lung": [
        "lung",
        "lungs",
        "pulmonary region",
        "pulmonary parenchyma",
        "lung fields",
        "lung tissue",
        "pulmonary tissue",
    ],
    "Trachea and Bronchie": [
        "trachea and bronchi",
        "central airways",
        "tracheobronchial tree",
        "large airways",
        "main airways",
        "trachea and bronchial tree",
        "proximal airways",
    ],
    "Mediastinum": [
        "mediastinum",
        "mediastinal region",
        "mediastinal structures",
        "central thoracic compartment",
        "middle thoracic compartment",
    ],
    "Heart": [
        "heart",
        "cardiac region",
        "cardiac structures",
        "myocardial region",
        "cardiac anatomy",
    ],
    "Esophagus": [
        "esophagus",
        "oesophagus",
        "esophageal region",
        "esophageal structure",
        "swallowing tube",
    ],
    "Pleura": [
        "pleura",
        "pleural space",
        "pleural region",
        "pleural lining",
        "pleural tissues",
        "pleural structures",
    ],
    "Bone": [
        "bone",
        "bones",
        "osseous structures",
        "bony thorax",
        "skeletal structures",
        "thoracic skeleton",
    ],
    "Thyroid": [
        "thyroid",
        "thyroid gland",
        "thyroid region",
        "thyroid tissue",
    ],
    "Breast": [
        "breast",
        "breasts",
        "breast tissue",
        "mammary tissue",
        "mammary region",
    ],
    "Abdomen": [
        "abdomen",
        "abdominal region",
        "abdominal cavity",
        "upper abdomen",
        "abdominal structures",
    ],
    "Others": [
        "other region",
        "other anatomical structures",
        "miscellaneous anatomy",
        "surrounding tissues",
        "other findings region",
    ],
    "Lower Thorax": [
        "lower thorax",
        "lower chest",
        "inferior thorax",
        "basal thorax",
        "lower thoracic region",
    ],
    "Liver and Biliary Tree": [
        "liver and biliary tree",
        "hepatobiliary system",
        "liver and bile ducts",
        "hepatic and biliary structures",
        "liver and biliary tract",
    ],
    "Gallbladder": [
        "gallbladder",
        "biliary bladder",
        "gallbladder region",
        "gallbladder structure",
    ],
    "Spleen": [
        "spleen",
        "splenic region",
        "splenic tissue",
        "splenic structure",
    ],
    "Pancreas": [
        "pancreas",
        "pancreatic region",
        "pancreatic tissue",
        "pancreatic structure",
    ],
    "Adrenal Glands": [
        "adrenal glands",
        "adrenals",
        "suprarenal glands",
        "adrenal region",
        "adrenal structures",
    ],
    "Kidneys and Ureters": [
        "kidneys and ureters",
        "renal system",
        "renal and ureteric structures",
        "upper urinary tract",
        "kidney and ureter region",
    ],
    "Gastrointestinal Tract": [
        "gastrointestinal tract",
        "GI tract",
        "digestive tract",
        "gastrointestinal system",
        "alimentary tract",
        "bowel region",
    ],
    "Peritoneal Cavity": [
        "peritoneal cavity",
        "peritoneal space",
        "abdominal peritoneal cavity",
        "peritoneal region",
    ],
    "Pelvic Organs": [
        "pelvic organs",
        "pelvic viscera",
        "organs in the pelvis",
        "pelvic structures",
        "pelvic cavity organs",
    ],
    "Vasculature": [
        "vasculature",
        "blood vessels",
        "vascular structures",
        "arterial and venous system",
        "vascular system",
    ],
    "Lymph Nodes": [
        "lymph nodes",
        "nodal structures",
        "lymphatic nodes",
        "lymph node regions",
    ],
    "Musculoskeletal": [
        "musculoskeletal system",
        "musculoskeletal structures",
        "bones and muscles",
        "muscles and skeleton",
        "musculoskeletal region",
    ],
    "Lungs and Airways": [
        "lungs and airways",
        "pulmonary system",
        "lung fields and airways",
        "respiratory structures",
        "pulmonary and airway region",
    ],
    "Cardiovascular": [
        "cardiovascular region",
        "cardiomediastinal silhouette",
        "heart and great vessels",
        "cardiac and vascular structures",
        "cardiovascular structures",
    ],
    "Hila and Mediastinum": [
        "hila and mediastinum",
        "hilar and mediastinal region",
        "mediastinal and hilar structures",
        "central thoracic structures",
        "hilar-mediastinal compartment",
    ],
    "Tubes and Catheters and Support Devices": [
        "tubes, catheters, and support devices",
        "support devices",
        "lines and tubes",
        "medical devices",
        "indwelling tubes and catheters",
        "support apparatus",
    ],
    "Musculoskeletal and Chest Wall": [
        "musculoskeletal system and chest wall",
        "chest wall and osseous structures",
        "bones and chest wall",
        "thoracic musculoskeletal structures",
        "bony thorax and chest wall",
    ],
    "Brain": [
        "brain",
        "intracranial structures",
        "cranial contents",
    ],
    "Spine": [
        "spine",
        "spinal column",
        "vertebral column",
        "spinal structures",
        "vertebral structures",
    ],
    "Abdominal": [
        "abdominal",
        "abdomen",
        "abdominal region",
        "upper abdomen",
        "subdiaphragmatic abdomen",
        "abdominal structures",
    ],
    "Other": [
        "other",
        "other region",
        "other anatomical structures",
        "miscellaneous findings region",
        "surrounding structures",
        "non-specific anatomy",
    ],
}

# Normalize keys so lookups with anatomy.lower() are always valid.
ANATOMY_SYNONYMS = {key.lower(): synonyms for key, synonyms in ANATOMY_SYNONYMS.items()}


def sample_anatomy_text(anatomy: str) -> str:
    key = anatomy.strip().lower()
    synonyms = ANATOMY_SYNONYMS.get(key)
    if synonyms:
        return random.choice(synonyms)
    # Fall back to raw text when a new anatomy label is unseen.
    if key not in _SEEN_UNKNOWN_ANATOMY:
        print(
            f"[dataset_utils] Unknown anatomy label encountered: '{anatomy}'. "
            "Falling back to raw label text."
        )
        _SEEN_UNKNOWN_ANATOMY.add(key)
    return anatomy.replace("_", " ").strip()
