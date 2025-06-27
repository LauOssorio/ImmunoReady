Disclaimer: this repo was created for learning and teaching purposes in the context of a Data Science bootcamp. Results have not scientif value and code is not peer-reviewed. Do not use for research

# ImmunoReady
This repository provides a machine learning-based pipeline to predict the immunogenicity of peptides for use in **cancer vaccine development**.

### Key Features
- Autoimmunity Risk Assessment: ImmunoReady is specifically trained to distinguish peptides that could potentially trigger autoimmune or inflammatory responses from those that are safe, based solely on their amino acid sequences.
- MHC/HLA Agnostic: The tool operates independently of MHC class and HLA type, allowing for broad applicability across different populations and facilitating generalization.
- Curated Training Data: Peptides used for training include known epitopes implicated in autoimmune and inflammatory diseases, as well as peptides derived from healthy tissues, ensuring robust discrimination between risky and safe candidates.
- Vaccine Development Support: By flagging high-risk peptides early in the design pipeline, ImmunoReady helps researchers avoid sequences that could lead to adverse effects, accelerating safe vaccine development.

### How It Works
ImmunoReady analyzes peptide sequences submitted by the user and predicts their safety profile. The underlying model has been trained on a diverse dataset:

- Risky: Peptides associated with known autoimmune and inflammatory disorders.
- Safe: Peptides derived from healthy human tissues, considered safe for use in vaccines.

### Use Cases
- Epitope Screening: Quickly evaluate candidate peptides for their likelihood to induce autoimmunity.
- Rational Vaccine Design: Select sequences most likely to be safe, reducing downstream risk and development costs.
- Immunology Research: Explore sequence patterns and risk factors associated with autoimmunity at the peptide level.

´´´
