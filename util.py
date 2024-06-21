# Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


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
    plt.plot(k_values, clusters_centers, 'o-', color='orange')
    plt.xlabel("Ilość klastrów (K)")
    plt.ylabel("Inercja klastera")
    plt.savefig('results/plots/elbow_plot.png', transparent = True)
    plt.title("Elbow Plot dla k-średnich")
    plt.show()

    # return clusters_centers, k_values


# Standarisation function
def standarisation_data(data_year):
    scaler = StandardScaler()
    scaler.fit(data_year)
    stand_data = scaler.transform(data_year)
    return stand_data


# KMeans Function
def kmeans_data(data_year, scaled_data, k, cluster_name):
    kmeans_model = KMeans(n_clusters=k, random_state=300)
    kmeans_model.fit(scaled_data, sample_weight=data_year['WEIGHT'])
    data_year[cluster_name] = kmeans_model.labels_


# EM Function
def em_data(data_year, scaled_data, k, cluster_name):
    gmm_model = GaussianMixture(n_components=k, random_state=300)
    gmm_model.fit(scaled_data)
    data_year[cluster_name] = gmm_model.predict(scaled_data)


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


# Description of the category of code_list
def descript(df, category):
    return df[df['Field'] == category].iloc[0]['Question/Description']


# Create dictionary based on code_list
def create_dictionary(df, category, exception=0):
    start_row = df[df['Field'] == category].index[0] + 1
    end_row = start_row
    while end_row < len(df) and pd.isna(df.iloc[end_row, 0]):
        end_row += 1
    category_df = df.iloc[start_row - exception:end_row, 1:3].dropna()
    category_df.columns = ['Question', 'Explanation']
    category_dict = pd.Series(category_df['Explanation'].values, index=category_df['Question']).to_dict()
    return category_dict


# Count mode of column
def mode(x):
    return pd.Series(x).value_counts().head(1).index[0]


# Count mode and print result
def mode_code(col_list, df_code, data_part, columns_null, columns_0, exe=False):
    col_dict = {}
    for col_name, num in col_list:
        col_name_2 = col_name
        if num != 0:
            col_name_2 = col_name[:-1] + str(int(col_name[-1:]) + num)
        elif exe:
            col_name_2 = 'Q4WIFI'

        if col_name in columns_null:
            indices = ~np.isnan(data_part[col_name])
            mode_res = create_dictionary(df_code, col_name_2, num)[mode(data_part[col_name][indices])]
        elif col_name in columns_0:
            indices = data_part[data_part[col_name] != 0].index
            mode_res = create_dictionary(df_code, col_name_2, num)[mode(data_part[col_name][indices])]
        else:
            mode_res = create_dictionary(df_code, col_name_2, num)[mode(data_part[col_name])]
        col_dict.update({col_name + '_mode': mode_res})
        print(f'Mode of {descript(df_code, col_name)}: {mode_res}')
    return col_dict


# Count mean (weighted, without null values) and print result
def mean_weight_nn(col_list, df_code, data_part, columns_null, columns_0):
    col_dict = {}
    for col_name in col_list:
        if col_name in columns_null:
            indices = ~np.isnan(data_part[col_name])
            result = round(np.average(data_part[col_name][indices], weights=data_part["WEIGHT"][indices]), 2)
        elif col_name in columns_0:
            indices = data_part[data_part[col_name] != 0].index
            result = round(np.average(data_part[col_name][indices], weights=data_part["WEIGHT"][indices]), 2)
        else:
            result = round(np.average(data_part[col_name], weights=data_part["WEIGHT"]), 2)
        print(f'Mean {descript(df_code, col_name)}: {result}')
        col_dict.update({col_name + '_mean': result})
    return col_dict


# Function to automate cluster analysis
def cluster_analysis(data_year, df_code, segment_name, cluster_num, columns_null, columns_0):
    data_part = data_year.loc[data_year[segment_name] == cluster_num]
    report = {}

    print('--- Cluster report ---\n')
    report.update({'Size': len(data_part)})
    print('Size of the cluster:', len(data_part))
    report.update(mean_weight_nn(['WEIGHT'], df_code, data_part, columns_null, columns_0))

    print('\n- Survey -')
    report.update(mode_code([('DAY', 0), ('METH', 0), ('SAQ', 0), ('LANG', 0)], df_code, data_part, columns_null, columns_0))

    print('\n- Flight -')
    report.update(mode_code([('STRATA', 0), ('PEAK', 0), ('AIRLINE_CODE', 0), ('DESTGEO', 0), ('DESTMARK', 0), ('Q2PURP1', 2), ('Q13COUNTY', 0)], df_code, data_part, columns_null, columns_0))
    report.update(mean_weight_nn(['HOWLONG'], df_code, data_part, columns_null, columns_0))

    print('\n- Transport -')
    report.update(mode_code([('Q3GETTO1', 2), ('Q3PARK', 0)], df_code, data_part, columns_null, columns_0))

    print('\n- Airports -')
    report.update(mode_code([('Q5TIMESFLOWN', 0), ('Q5FIRSTTIME', 0), ('Q6LONGUSE', 0), ('Q19Clear', 0), ('Q23FLY', 0), ('Q24SJC', 0), ('Q24OAK', 0)], df_code, data_part, columns_null, columns_0))

    print('\n- Use of services -')
    q4_names = list(filter(re.compile(r'Q4\w*').fullmatch, data_part.columns.to_list()))
    report.update(mode_code([(name, 0) for name in q4_names], df_code, data_part, columns_null, columns_0, True))
    report.update(mode_code([('Q11TSAPRE', 0), ('Q15PROBLEM', 0)], df_code, data_part, columns_null, columns_0))

    print('\n- Rating -')
    q7_names = list(filter(re.compile(r'Q7\w*').fullmatch, data_part.columns.to_list()))
    q7_rating = []
    for q7 in q7_names:
        indices = data_part[(data_part[q7] != 0) & (data_part[q7] != 6)].index
        q7_rating.append(round(np.average(data_part[q7][indices], weights=data_part['WEIGHT'][indices]), 2))
    print(f'Mean list of {descript(df_code, "Q7ALL")}: {q7_rating}')
    print(f'Mean of {descript(df_code, "Q7ALL")}: {round(np.average(q7_rating), 2)}')
    report.update({"Q7_list_mean": q7_rating, "Q7_mean": round(np.average(q7_rating), 2)})

    q9_names = list(filter(re.compile(r'Q9\w*').fullmatch, data_part.columns.to_list()))
    q9_rating = []
    for q9 in q9_names:
        indices = data_part[(data_part[q9] != 0) & (data_part[q9] != 6)].index
        q9_rating.append(round(np.average(data_part[q9][indices], weights=data_part['WEIGHT'][indices]), 2))
    q9_rating = [round(np.average(data_part[q9], weights=data_part['WEIGHT']), 2) for q9 in q9_names]
    print(f'Mean list of {descript(df_code, "Q9All")}: {q9_rating}')
    print(f'Mean of {descript(df_code, "Q9All")}: {round(np.average(q9_rating), 2)}')
    report.update({"Q9_list_mean": q9_rating, "Q9_mean": round(np.average(q9_rating), 2)})

    print(f'Mean of {descript(df_code, "Q10SAFE")}: {round(np.average(data_part["Q10Safe"], weights=data_part["WEIGHT"]), 2)}')
    print(f'Mode of {descript(df_code, "Q10SAFE")}: {mode(data_part["Q10Safe"])}')
    report.update({"Q10SAFE_mean": round(np.average(data_part["Q10Safe"], weights=data_part["WEIGHT"]), 2), "Q10SAFE_mode": mode(data_part["Q10Safe"])})
    report.update(mean_weight_nn(['Q12PRECHECKRATE', 'Q13GETRATE', 'Q14FIND', 'Q14PASSTHRU', 'NETPRO'], df_code,
                                 data_part, columns_null, columns_0))

    print('\n- Demographics -')
    report.update(mode_code([('Q17LIVE', 0), ('Q20Age', 0), ('Q21Gender', 0), ('Q22Income', 0)], df_code, data_part, columns_null, columns_0))
    report.update(mean_weight_nn(['Q20Age', 'Q22Income'], df_code, data_part, columns_null, columns_0))

    print('\n- Comments -')
    for col in ['Q8-1', 'Q9-1', 'Q10-1', 'Q12-1', 'Q15-1']:
        top_3 = data_part[data_part[col] != 0].groupby([col]).size().sort_values(ascending=False).index[0:3].to_list()
        report.update({col + '_com': top_3})
        print(f'{descript(df_code, col)}: {top_3}')

    report_df = pd.DataFrame(report.items(), columns=['Variable', 'Value'])
    return report_df
