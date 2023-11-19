import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "dataset/CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"
df = pd.read_csv(file_path)

df.head()
df.info()

# Replace a specific symbol with NaN (null value)
symbol_to_replace = '-'
df.replace(symbol_to_replace, pd.NA, inplace=True)

# Check for missing values
print(df.isnull().sum())

# List of columns to remove
columns_to_remove = ['service', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'tunnel_parents', 'detailed-label']
# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True)
print(df)

# Specify the column in which you want to replace null values with 0
column_to_replace = 'duration'

# Replace null values in the specified column with 0
df[column_to_replace].fillna(0, inplace=True)

# Specify the column based on which you want to remove duplicates
column_to_check_duplicates = 'uid'
# Drop duplicate rows based on the specified column
df.drop_duplicates(subset=column_to_check_duplicates, inplace=True)

# Check for an all-zero-values-column
column_to_check = 'missed_bytes'

# Identify cells where the value is not equal to 0
non_zero_values = df[df[column_to_check] != 0][column_to_check]

# Display or print the cells where the value is not equal to 0
print("Non-zero values in {}: \n{}".format(column_to_check, non_zero_values))

# Remove all zero valued column
df.drop(columns=[column_to_check], inplace=True)


# Select only the integer columns
integer_columns = df.select_dtypes(include='int')

# Calculate the correlation matrix for the integer columns
correlation_matrix = integer_columns.corr()

# Display the correlation matrix
print(correlation_matrix)

# Assuming 'ts' is the column with Unix timestamps
column_with_timestamps = 'ts'

# Convert Unix timestamps to human-readable datetime
df[column_with_timestamps] = pd.to_datetime(df[column_with_timestamps], unit='s', origin='unix')

# Sampling Example (Stratified):
X = df.drop(['label'], axis=1)
y = df['label']
X_sampled, _, y_sampled, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nShape of Sampled Data:")
print(X_sampled.shape)

df.to_csv('preprocessed_file.csv', index=False)

