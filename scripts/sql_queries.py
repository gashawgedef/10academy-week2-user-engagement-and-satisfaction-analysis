import pandas as pd
from sqlalchemy import create_engine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def execute_telecom_queries(db_url):
    engine = create_engine(db_url)

    # 1. Count of Unique IMSIs
    unique_imsi_count = pd.read_sql_query(
        """
        SELECT COUNT(DISTINCT "IMSI") AS unique_imsi_count
        FROM xdr_data;
    """,
        engine,
    )

    # 2. Average Duration of Calls
    average_duration = pd.read_sql_query(
        """
        SELECT AVG("Dur. (ms)") AS average_duration
        FROM xdr_data
        WHERE "Dur. (ms)" IS NOT NULL;
    """,
        engine,
    )

    # 3. Total Data Usage per User
    total_data_usage = pd.read_sql_query(
        """
        SELECT "IMSI", 
               SUM("Total UL (Bytes)") AS total_ul_bytes, 
               SUM("Total DL (Bytes)") AS total_dl_bytes
        FROM xdr_data
        GROUP BY "IMSI"
        ORDER BY total_dl_bytes DESC
        LIMIT 10;
    """,
        engine,
    )

    # 4. Average RTT by Last Location Name
    avg_rtt_by_location = pd.read_sql_query(
        """
        SELECT "Last Location Name", 
               AVG("Avg RTT DL (ms)") AS avg_rtt_dl, 
               AVG("Avg RTT UL (ms)") AS avg_rtt_ul
        FROM xdr_data
        GROUP BY "Last Location Name"
        HAVING COUNT(*) > 10
        ORDER BY avg_rtt_dl DESC;
    """,
        engine,
    )

    # Return results as a dictionary
    return {
        "unique_imsi_count": unique_imsi_count,
        "average_duration": average_duration,
        "total_data_usage": total_data_usage,
        "avg_rtt_by_location": avg_rtt_by_location,
    }


# Function to aggregate user data for Task 1.1
def aggregate_user_data(db_url):
    engine = create_engine(db_url)

    xdr_data = pd.read_sql_query(
        """
        SELECT "MSISDN/Number", "Dur. (ms)", "Total UL (Bytes)", "Total DL (Bytes)"
        FROM xdr_data;
    """,
        engine,
    )

    aggregated_data = (
        xdr_data.groupby("MSISDN/Number")
        .agg(
            num_sessions=("Dur. (ms)", "count"),
            total_duration=("Dur. (ms)", "sum"),
            total_ul_data=("Total UL (Bytes)", "sum"),
            total_dl_data=("Total DL (Bytes)", "sum"),
        )
        .reset_index()
    )

    aggregated_data["total_data"] = (
        aggregated_data["total_dl_data"] + aggregated_data["total_ul_data"]
    )
    return aggregated_data
# Function to perform descriptive statistics (EDA) for Task 1.2
def describe_data(db_url):
    user_data = aggregate_user_data(db_url)
    return user_data.describe()


# Function to create decile segmentation based on session duration
def segment_users_by_duration(db_url):
    user_data = aggregate_user_data(db_url)
    user_data["duration_decile"] = pd.qcut(
        user_data["total_duration"], 10, labels=False
    )
    return user_data.groupby("duration_decile").agg({"total_data": "sum"})


# Function to compute correlation matrix
def compute_correlation_matrix(db_url):
    user_data = aggregate_user_data(db_url)
    correlation_matrix = user_data[["total_duration", "total_data"]].corr()
    return correlation_matrix


# Function to perform PCA for dimensionality reduction
def perform_pca(db_url):
    user_data = aggregate_user_data(db_url)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(
        user_data[["total_duration", "total_data"]]
    )
    user_data["PC1"] = principal_components[:, 0]
    user_data["PC2"] = principal_components[:, 1]
    return user_data


# Function to perform K-means clustering for user engagement
def kmeans_clustering(db_url):
    user_data = aggregate_user_data(db_url)

    scaler = StandardScaler()
    user_data[["total_duration", "total_data"]] = scaler.fit_transform(
        user_data[["total_duration", "total_data"]]
    )

    kmeans = KMeans(n_clusters=3)
    user_data["cluster"] = kmeans.fit_predict(
        user_data[["total_duration", "total_data"]]
    )

    cluster_metrics = user_data.groupby("cluster").agg(
        {
            "total_duration": ["min", "max", "mean", "sum"],
            "total_data": ["min", "max", "mean", "sum"],
        }
    )

    return cluster_metrics


# Function to plot the distribution of session duration
def plot_session_duration_distribution(db_url):
    user_data = aggregate_user_data(db_url)
    sns.histplot(user_data["total_duration"], kde=True)
    plt.title("Session Duration Distribution")
    plt.show()


# Function to plot the correlation matrix
def plot_correlation_matrix(db_url):
    correlation_matrix = compute_correlation_matrix(db_url)
    sns.heatmap(correlation_matrix, annot=True)
    plt.title("Correlation Matrix for Application Traffic")
    plt.show()


# Function to plot PCA results
def plot_pca(db_url):
    user_data = perform_pca(db_url)
    sns.scatterplot(x=user_data["PC1"], y=user_data["PC2"])
    plt.title("PCA of User Data")
    plt.show()


# Function to plot K-means cluster results
def plot_kmeans_clusters(db_url):
    user_data = kmeans_clustering(db_url)
    sns.boxplot(x="cluster", y="total_duration", data=user_data)
    plt.title("Session Duration per Cluster")
    plt.show()
