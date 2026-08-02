# Visit with Us — Wellness Tourism MLOps Project

## Business Objective
Predict whether a customer will purchase the newly introduced Wellness Tourism Package before the marketing team contacts the customer.

## Repository Structure
```text
visit_with_us_mlops/
├── data/
│   └── tourism.csv
├── artifacts/
│   ├── train.csv
│   ├── test.csv
│   ├── experiment_results.csv
│   └── best_metrics.json
├── models/
│   └── best_model.joblib
├── src/
│   ├── data_validation.py
│   ├── prepare_data.py
│   └── train_model.py
├── .github/workflows/pipeline.yml
├── tourism_mlops_project.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## End-to-End Flow
1. Register and validate `data/tourism.csv`.
2. Clean the data and remove `Unnamed: 0` and `CustomerID`.
3. Split the data into stratified train/test workflow artifacts.
4. Tune multiple classification algorithms using GridSearchCV.
5. Compare F1 and ROC-AUC and save the best pipeline.
6. Run the Streamlit application using the committed model.
7. GitHub Actions automates validation, preparation, training, evaluation, artifact upload and model commit.

## Run Locally
```bash
pip install -r requirements.txt
python src/data_validation.py
python src/prepare_data.py
python src/train_model.py
streamlit run app.py
```

## Streamlit Community Cloud
Push this repository to GitHub, open Streamlit Community Cloud, select the repository and `app.py`, and deploy. The model is loaded from `models/best_model.joblib`.

## Evaluation
The notebook contains the data validation summary, cleaning, split, model experimentation, tuned parameters, evaluation metrics and final model path so that all required outputs are visible for assessment.
