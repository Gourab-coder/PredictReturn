# Product Return Prediction

A machine learning project that predicts whether an e-commerce order is likely to be returned. The project includes data cleaning, exploratory analysis, feature engineering, model comparison, threshold tuning, evaluation, and saved inference pipelines.

## Project Goal

Predict the binary `Returned` outcome:

- `0` = Not Returned
- `1` = Returned

The current pipeline uses 10 raw order features and selects the best-performing model using cross-validation and validation-tuned classification thresholds.

## Dataset

The training dataset is stored at [data/ecommerce_returns.csv](data/ecommerce_returns.csv).

| Column | Description |
|---|---|
| `Order_ID` | Unique order identifier; excluded from model training |
| `Product_Category` | Product category |
| `Product_Price` | Product unit price |
| `Quantity` | Number of units in the order |
| `Discount_Percentage` | Discount applied to the order |
| `Delivery_Days` | Number of days taken for delivery |
| `Payment_Method` | Payment method used |
| `Customer_Order_Count` | Number of previous orders by the customer |
| `Previous_Return_Count` | Number of previous returns by the customer |
| `Product_Rating` | Rating of the product |
| `Shipping_Type` | Delivery shipping type |
| `Returned` | Target variable |

An example order file for bulk inference is available at [data/new_orders.csv](data/new_orders.csv).

## Workflow

The main notebook is [notebooks/01_notebook_Product_Return_Prediction-v2.ipynb](notebooks/01_notebook_Product_Return_Prediction-v2.ipynb).

It performs the following steps:

1. Loads and validates the source CSV.
2. Removes duplicate records and invalid target values.
3. Converts invalid domain values to missing values.
4. Performs exploratory data analysis.
5. Creates a 70% training, 15% validation, and 15% test split.
6. Builds preprocessing and feature engineering inside scikit-learn pipelines.
7. Compares logistic regression, random forest, and histogram gradient boosting.
8. Uses 5-fold stratified cross-validation.
9. Tunes the prediction threshold on the validation set.
10. Evaluates the final models on the held-out test set.
11. Saves the best model and prediction artifacts.

## Feature Engineering

The pipeline automatically creates these additional features:

- `Customer_Return_Rate`
- `Effective_Unit_Price`
- `Total_Order_Value`
- `Discount_Category`
- `Customer_Type`

Feature engineering is fitted inside the model pipeline, so the saved model can accept the original 10 raw features directly.

## Model Results

The test-set comparison is saved in [models/model_evaluation.csv](models/model_evaluation.csv).

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.3631 | 0.5733 | 0.4769 | 0.8217 | 0.6035 | 0.6902 | 0.6085 |
| Logistic Regression | 0.3906 | 0.5733 | 0.4768 | 0.8169 | 0.6021 | 0.7120 | 0.6559 |
| Random Forest | 0.3584 | 0.5429 | 0.4585 | 0.8651 | 0.5993 | 0.6935 | 0.6104 |

**HistGradientBoosting** was selected as the saved model. Its tuned threshold is approximately `0.3631`.

The threshold was tuned to improve F1-score rather than to use an unadjusted `0.50` cutoff.

## Saved Artifacts

| File | Purpose |
|---|---|
| [models/return_prediction_model.joblib](models/return_prediction_model.joblib) | Complete scikit-learn pipeline |
| [models/model_metadata.json](models/model_metadata.json) | Selected model, threshold, features, and test metrics |
| [models/model_evaluation.csv](models/model_evaluation.csv) | Test-set model comparison |
| [models/bulk_return_predictions.csv](models/bulk_return_predictions.csv) | Predictions for `data/new_orders.csv` |

## Environment Setup

### Windows

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name predictreturn-venv --display-name "Python (PredictReturn venv)"
```

Open the notebook in VS Code or Jupyter and select:

```text
Python (PredictReturn venv)
```

## Run the Project

1. Confirm [data/ecommerce_returns.csv](data/ecommerce_returns.csv) exists.
2. Open [notebooks/01_notebook_Product_Return_Prediction-v2.ipynb](notebooks/01_notebook_Product_Return_Prediction-v2.ipynb).
3. Select the `Python (PredictReturn venv)` kernel.
4. Run every cell from top to bottom.
5. Review the cross-validation and test-set results.
6. Use the model or `models/bulk_return_predictions.csv` for inference.

## Making Bulk Predictions

Place new orders in [data/new_orders.csv](data/new_orders.csv) using the 10 raw feature columns. The notebook loads the saved pipeline and creates:

- `Return_Probability`
- `Predicted_Returned`
- `Prediction_Label`

Example input:

```csv
Order_ID,Product_Category,Product_Price,Quantity,Discount_Percentage,Delivery_Days,Payment_Method,Customer_Order_Count,Previous_Return_Count,Product_Rating,Shipping_Type
IND-ORD-200001,Footwear,789.0,3,24.8,3,Credit Card,6,1,4.2,Express
```

## Important Prediction Timing Caveat

The training data includes `Delivery_Days`, so this is currently a **post-delivery** model. It may not be suitable for predicting returns before an order is delivered unless delivery duration is estimated in advance.

For pre-dispatch prediction, remove or replace `Delivery_Days` and retrain the model with only features known before delivery.

## Future Improvements

- Implement XGBoost and other advanced classification algorithms.
- Build an interactive dashboard using Streamlit.
- Add real-time return-risk monitoring.
- Implement recommendations for return prevention and follow-up actions.
- Include customer history, product age, and delivery-service level features.
- Deploy the prediction model as a web application.
- Integrate a database for storing order and prediction records.

## Limitations

- The current dataset is synthetic.
- The model evaluation uses a random stratified train-validation-test split.
- The current model uses `Delivery_Days`, so it is a post-delivery prediction model.
- Threshold tuning prioritizes F1-score and may require adjustment for specific business costs.
- Prediction probability alone does not explain or guarantee the reason for a potential return.

## Author

**Gourab Das**

Machine Learning | Python | Data Science | Software Development

---

If you find this project useful, consider giving it a star!