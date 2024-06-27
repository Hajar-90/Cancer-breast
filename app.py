import tensorflow as tf

# Assuming cnn_model is your loaded Keras model
cnn_model = tf.keras.models.load_model('oneone.keras')

# Example of flattening output before dense layers
flatten_layer = tf.keras.layers.Flatten()(cnn_model.output)
dense_layer = tf.keras.layers.Dense(units=128, activation='relu')(flatten_layer)
output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_layer)

# Create a new model with modified layers
modified_model = tf.keras.Model(inputs=cnn_model.input, outputs=output_layer)
