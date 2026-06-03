import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_baseline_conv3d(input_channels=2, base=32):
    """
    Input: (T,H,W,C)
    Output: (T,H,W,1)
    """
    inp = keras.Input(shape=(None, None, None, input_channels))
    x = inp
    # shallow-ish 3D CNN
    x = layers.Conv3D(base, 3, padding="same", activation="relu")(x)
    x = layers.Conv3D(base, 3, padding="same", activation="relu")(x)
    x = layers.Conv3D(base*2, 3, padding="same", activation="relu")(x)
    x = layers.Conv3D(base*2, 3, padding="same", activation="relu")(x)
    x = layers.Conv3D(base, 3, padding="same", activation="relu")(x)
    out = layers.Conv3D(1, 1, padding="same", activation=None)(x)
    return keras.Model(inp, out, name="BaselineConv3D")
