from src.preprocess import load_data
from src.visualization import barplot_visualization, plot_heatmap
from src.clustering import cluster_data
from src.pca_autoencoder import apply_pca, build_autoencoder

def main():
    file_path = "data/sales_data_sample.csv"
    
    print("Loading data...")
    sales_df = load_data(file_path)
    
    print("Performing data visualization...")
    barplot_visualization(sales_df, "COUNTRY")
    plot_heatmap(sales_df)
    
    print("Applying clustering...")
    labels, centers = cluster_data(sales_df)
    
    print("Performing PCA...")
    pca_result = apply_pca(sales_df)
    
    print("Building and training autoencoder...")
    autoencoder, encoder = build_autoencoder(sales_df.shape[1])
    autoencoder.fit(sales_df, sales_df, batch_size=128, epochs=500, verbose=3)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
