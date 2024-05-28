
import csv
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from typing import List
import pathlib


def build_data(data_dir: pathlib.Path, filenames: List[str]):
    data_list = []
    for fname in filenames:
        with open(data_dir / 'training' / fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_list.append([float(row[1]), float(row[2])])
    return np.array(data_list)

# get the data
input_filenames: List[str] = \
    ['raw_mouse_events_70Alchs_Focused_cleaned.csv',
     'raw_mouse_events_70Alchs_Focused2_cleaned.csv',
     'raw_mouse_events_70Alchs_Focused3_cleaned.csv'
    ]
output_filename: str = 'learned_events_70Alchs_focused_standard.csv'

data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'
output_filepath: pathlib.Path = data_dir / 'generated' / output_filename

data = build_data(data_dir, input_filenames)
print(data[:5])

# Normalize the data
scaler = RobustScaler()
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

with open(output_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    for generated_row in generated_data:
        writer.writerow(list(generated_row))
