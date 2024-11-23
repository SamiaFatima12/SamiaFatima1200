import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('user_behaviour_dataset.csv')
# Sample data
data = {
    'user ID': [1, 2, 3, 4, 5],
    'Device Model': ['Google Pixel 5', 'OnePlus 9', 'Xiaomi Mi 11', 'Google Pixel 5', 'iPhone 12'],
    'Operating System': ['Android', 'Android', 'Android', 'Android', 'iOS'],
    'App Usage Time (min/day)': [393, 268, 154, 239, 187],
    'True Classification': ['TP', 'TP', 'FP', 'FP', 'FP']
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Check if the necessary columns are present
required_columns = ['Device Model', 'App Usage Time (min/day)', 'True Classification']
missing_columns = [col for col in required_columns if col not in df.columns]

# Renaming 'True Classification' to 'Actual_Classification' to use for accuracy
if 'Actual_Classification' not in df.columns:
    df['Actual_Classification'] = df['True Classification']

# Ensure 'App Usage Time (min/day)' is numeric
df['App Usage Time (min/day)'] = pd.to_numeric(df['App Usage Time (min/day)'], errors='coerce')

# Original classification function
def classify_usage_original(user_data):
    device = user_data['Device Model']
    usage_time = user_data['App Usage Time (min/day)']

    if device == "Google Pixel 5" and 100 <= usage_time <= 400:
        return "TP"
    elif device == "OnePlus 9" and 150 <= usage_time <= 300:
        return "TP"
    elif device == "Xiaomi Mi 11" and 120 <= usage_time <= 300:
        return "FP"
    elif device == "iPhone 12" and 80 <= usage_time <= 250:
        return "FP"
    else:
        return "FP"

# New classification function
def classify_usage_new(user_data):
    device = user_data['Device Model']
    usage_time = user_data['App Usage Time (min/day)']

    if device == "Google Pixel 5" and 320 <= usage_time <= 400:
        return "TP"
    elif device == "OnePlus 9" and 180 <= usage_time <= 290:
        return "TP"
    elif device == "Xiaomi Mi 11" and 100 <= usage_time <= 280:
        return "FP"
    elif device == "iPhone 12" and 90 <= usage_time <= 250:
        return "TP"
    else:
        return "FN"

    # User input to confirm before proceeding
user_input = input("Please enter the device model: ")
print(f"You entered: {user_input}")  # To display the input back to the user

# Apply classifications
df['Classification_Original'] = df.apply(classify_usage_original, axis=1)
df['Classification_New'] = df.apply(classify_usage_new, axis=1)

# Calculate accuracy
def calculate_accuracy(predictions, actual_labels):
    correct_predictions = (predictions == actual_labels).sum()
    total_predictions = len(actual_labels)
    return correct_predictions / total_predictions

# Accuracy for both classifications
accuracy_original = calculate_accuracy(df['Classification_Original'], df['Actual_Classification'])
accuracy_new = calculate_accuracy(df['Classification_New'], df['Actual_Classification'])

print(f"Accuracy of Original Classification: {accuracy_original * 100:.2f}%")
print(f"Accuracy of New Classification: {accuracy_new * 60:.2f}%")

# Print the DataFrame
print("\nDataFrame with Both Classifications:")
print(df)

# Assuming you have already calculated accuracy_original and accuracy_new

# Create a bar graph
classifications = ['Original', 'New']
accuracies = [accuracy_original, accuracy_new]

plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.bar(classifications, accuracies, color=['blue', 'orange'])  # Customize colors

# Add labels, title, and accuracy values on top of bars
plt.xlabel('Classification Method')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')  # Adjust text position as needed

plt.ylim(0, 1.1)  # Set y-axis limits for better visualization
plt.grid(True)
# To see the output, run the code.