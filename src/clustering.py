from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

def cluster_data(df, num_clusters=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(num_clusters)
    kmeans = kmeans.fit(df_scaled)
    labels = kmeans.labels_
    
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=[df.columns])
    cluster_centers = scaler.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(cluster_centers, columns=[df.columns])
    
    return labels, cluster_centers
