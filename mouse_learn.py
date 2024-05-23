import csv
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler

#def learn():


# get the data
input_filename: str = 'raw_mouse_events_cleaned.csv' #args[0]
output_filename: str = 'learned_events.csv'

data_list = []
with open(input_filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data_list.append([float(row[1]), float(row[2])])
print(data_list[:5])
data = np.array(data_list)
print(data.shape)

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)


# Define the autoencoder
input_dim = 2
latent_dim = 3

input_layer = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Define the decoder model
encoded_input = Input(shape=(latent_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data, data, epochs=100, batch_size=32, shuffle=True)

# Generate new data
latent_space_samples = np.random.normal(size=(len(data), latent_dim))
generated_data = decoder.predict(latent_space_samples)
print(generated_data)