
import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')

# Display basic information about the dataset
print('Dataset Info:')
df.info()

# Display descriptive statistics
print('\nDescriptive Statistics:')
print(df.describe())

# Display the first 5 rows
print('\nFirst 5 Rows:')
print(df.head())

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Save a summary to a text file
with open('data_summary.txt', 'w') as f:
    f.write('Dataset Info:\n')
    df.info(buf=f)
    f.write('\nDescriptive Statistics:\n')
    f.write(df.describe().to_string())
    f.write('\nFirst 5 Rows:\n')
    f.write(df.head().to_string())
    f.write('\nMissing Values:\n')
    f.write(df.isnull().sum().to_string())


