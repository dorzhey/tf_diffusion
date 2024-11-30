import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

file_dir = os.path.dirname(__file__)
save_path = os.path.join(file_dir, 'generated_data')

df = pd.read_csv(save_path + "/sampled_df_columns_renamed.csv", index_col=0)


import re



def get_bin_medians(data: pd.DataFrame):
    # Calculate bin edges using qcut
    _, bins = pd.qcut(data[data['cre_expression'] > 0]['cre_expression'], 9, retbins=True)

    # Assign each row to a bin using cut
    data['cre_expression_discrete'] = pd.cut(data['cre_expression'], bins=bins, include_lowest=True)

    # Calculate the median for each bin
    bin_medians = data[data['cre_expression'] > 0].groupby(by = 'cre_expression_discrete', observed=False)['cre_expression'].median()

    # Replace bin labels with the median value of the bin
    data['cre_expression_discrete'] = data['cre_expression_discrete'].map(bin_medians)

    # Convert the column to a numeric type before filling NaN values
    data['cre_expression_discrete'] = pd.to_numeric(data['cre_expression_discrete'], errors='coerce')

    # Replace NaN values (for zero expressions) with 0
    data['cre_expression_discrete'] = data['cre_expression_discrete'].fillna(0)

    return data['cre_expression_discrete']
    


df['cre_expression_discrete'] = get_bin_medians(df)




# Group by tf_cluster and cre_expression and collect unique strings in cre_sequence
grouped = df.groupby(['tf_cluster', 'cre_expression'])['cre_sequence'].unique().reset_index()

# Create a dictionary to store the sets of unique strings for each group
group_string_sets = {}

# Fill the dictionary with sets of unique strings for each group
for name, group in grouped.groupby(['tf_cluster', 'cre_expression']):
    group_string_sets[name] = set(group['cre_sequence'].explode())

# Create a list of unique group names
unique_groups = list(group_string_sets.keys())

# Initialize a matrix to store similarity results
similarity_matrix = np.zeros((len(unique_groups), len(unique_groups)))

# Function to calculate Jaccard similarity index
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Calculate similarity between the groups and fill the matrix
for i, group1 in enumerate(unique_groups):
    for j, group2 in enumerate(unique_groups):
        similarity_matrix[i, j] = jaccard_similarity(group_string_sets[group1], group_string_sets[group2])

# Create a DataFrame for the heatmap
similarity_df = pd.DataFrame(similarity_matrix, index=unique_groups, columns=unique_groups)

print("creating figure")
# Plot the heatmap
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Jaccard Similarity Index Between Groups")
plt.xlabel("Group")
plt.ylabel("Group")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig("generated_data/label_heatmap_jaccard.png")