# Step 1: Import necessary libraries
import pandas as pd  # Import the pandas library, which is used for data manipulation and analysis
import seaborn as sns  # Import the seaborn library, which is used for statistical data visualization
import matplotlib.pyplot as plt  # Import the matplotlib library, specifically the pyplot module, for creating plots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
 
# Step 2: Read data from a CSV file
# The pd.read_csv() function reads a CSV file and loads it into a DataFrame, which is a table-like data structure
df = pd.read_csv("troop_movements.csv")
 
# Step 3: Create a new column based on a condition
# We are adding a new column called 'is_resistance' to the DataFrame
# The .apply() method is used to apply a function to each element in the 'empire_or_resistance' column
# The lambda function checks if the value is 'Resistance' and returns True if it is, otherwise False
df['is_resistance'] = df['empire_or_resistance'].apply(lambda x: x == 'resistance')
 
# Step 4: Group data and count occurrences
# Group the data by the 'empire_or_resistance' column and count how many times each value appears
# The .reset_index() method is used to convert the result into a DataFrame with a column named 'Count'
grouped_counts_side = df.groupby('empire_or_resistance').size().reset_index(name='Count')
 
# Group the data by the 'homeworld' column and count how many times each value appears
# Again, use .reset_index() to create a DataFrame with a column named 'Count'
grouped_counts_homeworld = df.groupby('homeworld').size().reset_index(name='Count')
 
# Group the data by the 'unit_type' column and count how many times each value appears
# Use .reset_index() to create a DataFrame with a column named 'Count'
grouped_counts_unit_type = df.groupby('unit_type').size().reset_index(name='Count')
 
# Step 5: Display data to verify changes
# Print the first few rows of the DataFrame to check if the new column 'is_resistance' was added correctly
print("Data with 'is_resistance' feature:")
print(df.head())
 
# Print the counts of units grouped by 'empire_or_resistance' with column titles
print("\nCounts of units by side (Empire vs Resistance):")
print(grouped_counts_side)
 
# Print the counts of characters grouped by 'homeworld' with column titles
print("\nCounts of characters by homeworld:")
print(grouped_counts_homeworld)
 
# Print the counts of characters grouped by 'unit_type' with column titles
print("\nCounts of characters by unit type:")
print(grouped_counts_unit_type)
 
# Step 6: Visualize data using Seaborn
# Create a count plot to visualize the distribution of units between Empire and Resistance
sns.countplot(data=df, x='empire_or_resistance', hue='empire_or_resistance', palette=['#1f77b4', '#ff7f0e'], legend=False)
 
# Set the title and labels for the plot to make it more informative
plt.title('Distribution of Units: Empire vs Resistance')
plt.xlabel('Side')
plt.ylabel('Count')
 
# Display the plot on the screen
plt.show()
 
 
# Step 7: Prepare the Data
# use pd.get_dummies() to convert categorical variables into numerical format
# This is necessary for machine learning models, which require numerical input
# 'homeworld' and 'unit_type' are the features we are encoding
X_encoded = pd.get_dummies(df[['homeworld', 'unit_type']])
 
# 'empire_or_resistance' is the target variable we want to predict
y = df['is_resistance']
 
 
# Step 8: Split the Data
# We split the data into training and testing sets using train_test_split()
# X_encoded is the feature set, y is the target variable
# test_size=0.2 means 20% of the data will be used for testing, and 80% for training
# random_state=42 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.8, random_state=42)
 
# Step 9: Train the Model
# We create a DecisionTreeClassifier model, which is a type of machine learning model
# random_state=42 ensures reproducibility of the model's behavior
model = DecisionTreeClassifier(random_state=42)
 
# We fit the model using the training data (X_train and y_train)
model.fit(X_train, y_train)
 
# Step 10: Evaluate the Model
# We use the model to predict the target variable for the test data (X_test)
y_pred = model.predict(X_test)
 
# We calculate the accuracy of the model using accuracy_score()
# accuracy_score compares the predicted values (y_pred) with the actual values (y_test)
accuracy = accuracy_score(y_test, y_pred)
 
# Print the accuracy of the model, formatted to two decimal places
print(f"Model Accuracy: {accuracy:.2f}")
 
# Step 11: Feature Importance
# We extract the importance of each feature using model.feature_importances_
importances = model.feature_importances_
 
# We create a DataFrame to display the feature names and their importance
feature_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
 
# Step 12: Visualize Feature Importance
# We sort the feature_importances DataFrame by the 'Importance' column in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances.to_csv("feature_importances.csv", index=False)

# Create a bar plot to visualize the importance of each feature
# sns.barplot() creates a bar plot with 'Feature' on the x-axis and 'Importance' on the y-axis
sns.barplot(data=feature_importances, x='Feature', y='Importance', palette='viridis')
 
# Set the title and labels for the plot to make it more informative
plt.title('Feature Importance in Predicting Empire vs Resistance')
plt.xlabel('Feature')
plt.ylabel('Importance')
 
# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, ha='right')
 
# Adjust the layout to prevent clipping of labels and display the plot
plt.tight_layout()
plt.savefig("feature_importance_plot.png", bbox_inches='tight')
plt.show()
 
import pickle
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
 
print("Model saved to trained_model.pkl")
