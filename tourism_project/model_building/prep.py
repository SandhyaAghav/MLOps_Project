# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Sandhya-2025/tourism-package-purchase/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# Define the target variable for the classification task
target = 'ProdTaken'

# Replacing Fe male with Female
tourism_dataset["Gender"].replace("Fe Male", "Female", inplace=True)

ordinal_cols = [
    "CityTier",
    "PreferredPropertyStar",
    "PitchSatisfactionScore"
]
binary_cols = ["Passport", "OwnCar"]
ordinal_encoder = OrdinalEncoder()

# List of numerical features in the dataset
numeric_features = [
    'Age',             # Age of the customer.
    'MonthlyIncome',   # Gross monthly income of the customer.
    'DurationOfPitch', # Duration of the sales pitch delivered to the customer.
    'NumberOfTrips'    # Average number of trips the customer takes annually.
    'NumberOfPersonVisiting',   #Total number of people accompanying the customer on the trip.
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer.
    'NumberOfFollowups'         # Total number of follow-ups by the salesperson after the sales pitch.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',     # The method by which the customer was contacted (Company Invited or Self Inquiry).
    'Occupation',        # Customer's occupation (e.g., Salaried, Freelancer).
    'Gender',            # Gender of the customer (Male, Female).
    'MaritalStatus',     # Marital status of the customer (Single, Married, Divorced).
    'Designation',       # Customer's designation in their current organization.
    'ProductPitched',    # The type of product pitched to the customer.
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features + ordinal_cols + binary_cols]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Sandhya-2025/tourism-package-purchase",
        repo_type="dataset",
    )
