Heritage Damage — Enhanced Streamlit Deployment App
==================================================

Files:
- heritage_deploy_app.py  : main Streamlit app (single-file)
- requirements.txt        : Python package list (includes optional libs)
- README.md               : this file

Quick start (local):
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Run app:
   streamlit run heritage_deploy_app.py

Uploading data:
- Upload 'building_data' CSV/XLSX with columns: building_id, location_lat, location_lon, age, material_type (distance_to_epicenter optional)
- Upload 'damage_reports' CSV/XLSX with columns: building_id, damage_level (labels: minor/moderate/major or variants)
- Optionally upload 'earthquake_data' CSV/XLSX containing epicenter coordinates (epicenter_lat/epicenter_lon or latitude/longitude)

Notes:
- Optional features (LightGBM, CatBoost, NGBoost, SHAP, SMOTETomek) are enabled only when those packages are installed.
- For production, consider using a container (Dockerfile) or Streamlit Cloud. Reduce n_estimators on LightGBM/CatBoost for faster runs.
- Validate model performance and calibration before making operational decisions.

