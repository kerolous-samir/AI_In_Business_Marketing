import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA

def apply_pca(df, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)
    return pca_result

def build_autoencoder(input_dim):
    input_df = Input(shape=(input_dim,))
    x = Dense(50, activation='relu')(input_df)
    x = Dense(500, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    encoded = Dense(8, activation='relu')(x)
    x = Dense(2000, activation='relu')(encoded)
    x = Dense(500, activation='relu')(x)
    decoded = Dense(input_dim)(x)

    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder, encoder
