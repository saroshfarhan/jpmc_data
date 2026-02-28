# Task 3: Credit risk analysis

## Here is the background information on your task

You have now moved to a new team assisting the retail banking arm, which has been experiencing higher-than-expected default rates on personal loans. Loans are an important source of revenue for banks, but they are also associated with the risk that borrowers may default on their loans. A default occurs when a borrower stops making the required payments on a debt.

The risk team has begun to look at the existing book of loans to see if more defaults should be expected in the future and, if so, what the expected loss will be. They have collected data on customers and now want to build a predictive model that can estimate the probability of default based on customer characteristics. A better estimate of the number of customers defaulting on their loan obligations will allow us to set aside sufficient capital to absorb that loss. They have decided to work with you in the QR team to help predict the possible losses due to the loans that would potentially default in the next year.

Charlie, an associate in the risk team, who has been introducing you to the business area, sends you a small sample of their loan book and asks if you can try building a prototype predictive model, which she can then test and incorporate into their loss allowances.

## Here is your task

The risk manager has collected data on the loan borrowers. The data is in tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics. There is also a column indicating if the borrower has previously defaulted on a loan. You must use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). Use the provided data to train a function that will estimate the probability of default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

- You should produce a function that can take in the properties of a loan and output the expected loss.
- You can explore any technique ranging from a simple regression or a decision tree to something more advanced. You can also use multiple methods and provide a comparative analysis.

---

## Findings

### Dataset
- 10,000 borrower records with 6 features and a binary `default` label
- Class distribution: ~82% non-default, ~18% default (moderate imbalance)
- No missing values or duplicate records

### EDA
- **Strongest predictors of default:**
  - `credit_lines_outstanding` (correlation: +0.86) — most dominant feature
  - `total_debt_outstanding` (correlation: +0.76)
  - `fico_score` (correlation: -0.32) — higher score reduces default risk
  - `years_employed` (correlation: -0.28) — more stability reduces risk
- **Multicollinearity noted:**
  - `credit_lines_outstanding` ↔ `total_debt_outstanding`: 0.85
  - `loan_amt_outstanding` ↔ `income`: 0.84
- Outliers present in `loan_amt_outstanding`, `total_debt_outstanding`, and `income` — consistent with a synthetic dataset

### Model — Logistic Regression
- Stratified 60/20/20 train/val/test split to preserve class balance
- `class_weight='balanced'` used to handle the 82/18 imbalance
- Features scaled with `StandardScaler` prior to fitting

**Results:**

| Split | ROC-AUC | Accuracy |
|-------|---------|----------|
| Val   | 1.000   | 99%      |
| Test  | 1.000   | 100%     |

- Perfect separation achieved — consistent with the synthetic nature of the dataset
- No hyperparameter tuning required given perfect AUC

### Expected Loss Function
- Implemented in `task3.py` as `expected_loss(...)`
- Formula: `EL = PD × loan_amt_outstanding × (1 - recovery_rate)`
- Recovery rate assumed at **10%**
- Uses raw `predict_proba` output (no threshold applied) for continuous EL estimation

**Sample outputs:**

| Borrower Profile | PD | Expected Loss |
|------------------|----|--------------|
| Low risk (0 credit lines, FICO 750, income $90k) | 0.00% | $0.00 |
| Medium risk (2 credit lines, FICO 600, income $55k) | 0.21% | $9.58 |
| High risk (4 credit lines, FICO 500, income $30k) | 100.00% | $7,200.00 |