# K-Means Customer Segmentation (Deliverable)

This repository contains a ready-to-run template for performing K-Means clustering on an online-retail dataset and producing outputs suitable for stakeholder reporting.

## Contents
- `kmeans_clustering.py` - Main script to run preprocessing, elbow method, K-Means, PCA scatter, and save outputs.
- `report.md` - Business-style editable report with executive summary, findings template and recommendations.
- `requirements.txt` - Python package requirements.
- `run_example.sh` - Example commands to run the pipeline.
- `outputs/` - Directory (created at runtime) with saved figures and CSVs.

## How to run
1. Create a Python virtual environment and install requirements:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Place your dataset CSV (for example `online_retail.csv`) in the project folder.
   - If your dataset has an identifier column like `CustomerID`, pass it with `--id_col`.
3. Run the script:
    ```bash
    python kmeans_clustering.py --input online_retail.csv --id_col CustomerID --output_dir outputs --max_k 8
    ```
4. Check the `outputs/` folder for `elbow_wcss.png`, `clusters_pca.png`, `clustered_data.csv`, `summary.json`, and `cluster_profiles.json`.
