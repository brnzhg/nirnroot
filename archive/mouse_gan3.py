import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
import tensorflow as tf
from keras import layers

import pathlib
import csv
from typing import List


def build_data(data_dir: pathlib.Path, filenames: List[str]):
    data_list = []
    for fname in filenames:
        with open(data_dir / 'training' / fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_list.append([float(row[1]), float(row[2])])
    return np.array(data_list)

input_filenames: List[str] = \
    ['bz_constant_050124.csv',
     'bz_constant_052324.csv',
     'bz_constant_052324_2.csv',
    ]
output_filename: str = 'bz_constant_gan.csv'

data_dir: pathlib.Path = pathlib.Path.cwd() / 'tempdata'
output_filepath: pathlib.Path = data_dir / 'generated' / output_filename

data = build_data(data_dir, input_filenames)



# Handle outliers with RobustScaler
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data)

# Handle skewness with PowerTransformer (Yeo-Johnson by default)
transformer = PowerTransformer(method='yeo-johnson')
data_transformed = transformer.fit_transform(data_scaled)

# Define the generator with increased capacity
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_dim=2),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softplus')  # Ensure outputs are nonnegative
    ])
    return model

# Define the discriminator with increased capacity
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Combined GAN model (generator + discriminator)
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Function to train the GAN
def train_gan(gan, data, epochs=3000, batch_size=4):
    generator, discriminator = gan.layers
    for epoch in range(epochs):
        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, 2))
        generated_data = generator.predict(noise)

        # Get a random batch of real data
        real_data = data[np.random.randint(0, data.shape[0], batch_size)]
        
        # Create labels
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        
        # Train the discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(real_data, labels_real)
        discriminator.train_on_batch(generated_data, labels_fake)
        
        # Train the generator via the GAN model
        noise = np.random.normal(0, 1, (batch_size, 2))
        discriminator.trainable = False
        gan.train_on_batch(noise, labels_real)

    return gan

# Train the GAN
trained_gan = train_gan(gan, data_transformed)

# Generate new data
noise = np.random.normal(0, 1, (len(data), 2))
generated_data = generator.predict(noise)

# Inverse transform the generated data
generated_data_inverse_transformed = transformer.inverse_transform(generated_data)
generated_data_denormalized = scaler.inverse_transform(generated_data_inverse_transformed)
new_data = generated_data_denormalized.tolist()
print(new_data)

# Compare variances
original_variance = np.var(data, axis=0)
generated_variance = np.var(new_data, axis=0)
print("Original Variance:", original_variance)
print("Generated Variance:", generated_variance)


print(new_data[:5])
with open(output_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    for generated_row in new_data:
        writer.writerow(list(generated_row))