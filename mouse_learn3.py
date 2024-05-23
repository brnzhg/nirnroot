
import csv
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

#=====================
# Define the autoencoder
input_dim = data_normalized.shape[1]  # This should be 2
latent_dim = 3  # Arbitrary latent dimension, can be tuned

input_layer = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_layer)
decoded = Dense(input_dim)(encoded)  # Linear activation function

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Define the decoder model
encoded_input = Input(shape=(latent_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(data_normalized, data_normalized, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Generate new data
latent_space_samples = np.random.normal(size=(len(data_normalized), latent_dim))
generated_data_normalized = decoder.predict(latent_space_samples)

# De-normalize the generated data
generated_data = scaler.inverse_transform(generated_data_normalized)

# Check the shape and first few rows of the generated data
print("Generated data shape:", generated_data.shape)
print("First few rows of generated data:\n", generated_data[:5])

with open(output_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    for generated_row in generated_data:
        writer.writerow(list(generated_row))
