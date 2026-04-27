import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

N = 7043  # mirrors real IBM Telco size

def generate_churn_dataset(n=N, save_path="data/telco_churn.csv"):
    tenure         = np.random.randint(1, 73, n)
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges  = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges  = np.clip(total_charges, 0, None)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.24, 0.21]
    )
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    phone_service   = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines  = np.where(phone_service == "No", "No phone service",
                        np.random.choice(["Yes", "No"], n, p=[0.48, 0.52]))
    online_security = np.where(internet_service == "No", "No internet service",
                        np.random.choice(["Yes", "No"], n, p=[0.29, 0.71]))
    tech_support    = np.where(internet_service == "No", "No internet service",
                        np.random.choice(["Yes", "No"], n, p=[0.29, 0.71]))
    streaming_tv    = np.where(internet_service == "No", "No internet service",
                        np.random.choice(["Yes", "No"], n, p=[0.38, 0.62]))
    streaming_movies = np.where(internet_service == "No", "No internet service",
                        np.random.choice(["Yes", "No"], n, p=[0.39, 0.61]))
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner         = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents      = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])

    # Churn probability driven by real-world signals
    churn_score = (
        0.35 * (contract == "Month-to-month").astype(float)
        - 0.25 * (contract == "Two year").astype(float)
        + 0.20 * (internet_service == "Fiber optic").astype(float)
        - 0.15 * (online_security == "Yes").astype(float)
        - 0.15 * (tech_support == "Yes").astype(float)
        + 0.15 * (payment_method == "Electronic check").astype(float)
        - 0.20 * (tenure / 72)
        + 0.10 * (monthly_charges / 120)
        + np.random.normal(0, 0.15, n)
    )
    # Shift to target ~26% churn rate (realistic for telecom)
    churn_prob = 1 / (1 + np.exp(-3 * (churn_score - 0.55)))
    churn      = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    customer_ids = [f"CUS-{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "customer_id":       customer_ids,
        "senior_citizen":    senior_citizen,
        "partner":           partner,
        "dependents":        dependents,
        "tenure":            tenure,
        "phone_service":     phone_service,
        "multiple_lines":    multiple_lines,
        "internet_service":  internet_service,
        "online_security":   online_security,
        "tech_support":      tech_support,
        "streaming_tv":      streaming_tv,
        "streaming_movies":  streaming_movies,
        "contract":          contract,
        "paperless_billing": paperless_billing,
        "payment_method":    payment_method,
        "monthly_charges":   monthly_charges,
        "total_charges":     total_charges,
        "churn":             churn,
    })

    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}  |  shape: {df.shape}  |  churn rate: {churn.mean():.1%}")
    return df


if __name__ == "__main__":
    generate_churn_dataset()
