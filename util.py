# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Elbow function
def find_best_clusters(df, maximum_K):

    clusters_centers = []
    k_values = []

    for k in range(1, maximum_K):

        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)

    # Generate elbow plot
    plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Ilość klastrów (K)")
    plt.ylabel("Inercja klastera")
    plt.savefig('elbow_plot.png', transparent = True)
    plt.title("Elbow Plot dla k-średnich")
    plt.show()

    # return clusters_centers, k_values


# Standarisation function
def standarisation_data(data_year):
    scaler = StandardScaler()
    scaler.fit(data_year)
    scaled_data = scaler.transform(data_year)
    return scaled_data


# KMeans Function
def kmeans_data(data_year, scaled_data, k, cluster_name):
    kmeans_model = KMeans(n_clusters=k, random_state=300)
    kmeans_model.fit(scaled_data, sample_weight=data_year['WEIGHT'])
    data_year[cluster_name] = kmeans_model.labels_


# Used feature in tree
def used_feature(_tree, tree, feature_names):
    feature = []
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            feature.append(name)
            recurse(tree_.children_left[node], depth + 1)
            recurse(tree_.children_right[node], depth + 1)

    recurse(0, 1)
    return feature


# Create dictionary based on code list
def create_dictionary(df, category, exception=0):
    start_row = df[df['Field'] == category].index[0] + 1
    end_row = start_row
    while end_row < len(df) and pd.isna(df.iloc[end_row, 0]):
        end_row += 1
    category_df = df.iloc[start_row - exception:end_row, 1:3].dropna()
    category_df.columns = ['Question', 'Explanation']
    category_dict = pd.Series(category_df['Explanation'].values, index=category_df['Question']).to_dict()
    return category_dict
