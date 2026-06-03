import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .physics_metrics import wave_residual_norm
from .models_sacu import stitch_patches

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-12)

@tf.function
def compute_influence_weights(patch_outs, regions, y_true, beta=5.0):
    """
    Compute e_i per patch region and return softmax weights.
    e_i = mean |patch - truth_region|
    """
    B = tf.shape(y_true)[0]
    N = len(patch_outs)
    errs = []
    for i, (y0,y1,x0,x1) in enumerate(regions):
        yt = y_true[:, :, y0:y1, x0:x1, :]
        yp = patch_outs[i]
        e = tf.reduce_mean(tf.abs(yp - yt), axis=[1,2,3,4])  # (B,)
        errs.append(e)
    E = tf.stack(errs, axis=1)  # (B,N)
    w = tf.nn.softmax(-beta * E, axis=1)
    return w, E

class Trainer:
    def __init__(self, model, optimizer, run_dir,
                 use_physics_loss=False, lambda_phys=0.1, beta=5.0):
        self.model = model
        self.opt = optimizer
        self.run_dir = run_dir
        self.use_physics_loss = use_physics_loss
        self.lambda_phys = lambda_phys
        self.beta = beta

        self.train_loss = keras.metrics.Mean(name="train_loss")
        self.train_mae  = keras.metrics.Mean(name="train_mae")
        self.train_rmse = keras.metrics.Mean(name="train_rmse")
        self.train_res  = keras.metrics.Mean(name="train_residual")

        self.val_mae  = keras.metrics.Mean(name="val_mae")
        self.val_rmse = keras.metrics.Mean(name="val_rmse")
        self.val_res  = keras.metrics.Mean(name="val_residual")

    @tf.function
    def train_step_baseline(self, x, y_true, c_field, dt, dx):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_pred - y_true))
            if self.use_physics_loss:
                res = wave_residual_norm(y_pred, c_field, dt, dx)
                loss = loss + self.lambda_phys * res
            else:
                res = wave_residual_norm(y_pred, c_field, dt, dx)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, mae(y_true, y_pred), rmse(y_true, y_pred), res

    @tf.function
    def train_step_sacu(self, x, y_true, c_field, dt, dx):
        with tf.GradientTape() as tape:
            _, aux = self.model(x, training=True)
            patch_outs = aux["patch_outs"]
            regions = aux["regions"]
            H = tf.shape(x)[2]
            W = tf.shape(x)[3]

            w, _ = compute_influence_weights(patch_outs, regions, y_true, beta=self.beta)
            y_pred = stitch_patches(patch_outs, regions, w, H, W)

            loss = tf.reduce_mean(tf.square(y_pred - y_true))
            res = wave_residual_norm(y_pred, c_field, dt, dx)
            if self.use_physics_loss:
                loss = loss + self.lambda_phys * res

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, mae(y_true, y_pred), rmse(y_true, y_pred), res

    @tf.function
    def eval_step(self, x, y_true, c_field, dt, dx, is_sacu: bool):
        if not is_sacu:
            y_pred = self.model(x, training=False)
        else:
            _, aux = self.model(x, training=False)
            patch_outs = aux["patch_outs"]
            regions = aux["regions"]
            H = tf.shape(x)[2]
            W = tf.shape(x)[3]
            w, _ = compute_influence_weights(patch_outs, regions, y_true, beta=self.beta)
            y_pred = stitch_patches(patch_outs, regions, w, H, W)

        m = mae(y_true, y_pred)
        r = rmse(y_true, y_pred)
        res = wave_residual_norm(y_pred, c_field, dt, dx)
        return m, r, res

    def fit(self, train_ds, val_ds, epochs, is_sacu: bool):
        for ep in range(1, epochs+1):
            self.train_loss.reset_state()
            self.train_mae.reset_state()
            self.train_rmse.reset_state()
            self.train_res.reset_state()

            for batch in train_ds:
                features, y_true = batch
                x = features["x"]
                c = features["c_field"]
                dt = features["dt"]
                dx = features["dx"]

                if is_sacu:
                    loss, m, r, res = self.train_step_sacu(x, y_true, c, dt, dx)
                else:
                    loss, m, r, res = self.train_step_baseline(x, y_true, c, dt, dx)

                self.train_loss.update_state(loss)
                self.train_mae.update_state(m)
                self.train_rmse.update_state(r)
                self.train_res.update_state(res)

            # validation
            self.val_mae.reset_state()
            self.val_rmse.reset_state()
            self.val_res.reset_state()

            for batch in val_ds:
                features, y_true = batch
                x = features["x"]
                c = features["c_field"]
                dt = features["dt"]
                dx = features["dx"]
                m, r, res = self.eval_step(x, y_true, c, dt, dx, is_sacu=is_sacu)
                self.val_mae.update_state(m)
                self.val_rmse.update_state(r)
                self.val_res.update_state(res)

            print(
                f"Epoch {ep}/{epochs} | "
                f"loss {self.train_loss.result().numpy():.5f} | "
                f"MAE {self.train_mae.result().numpy():.5f} | "
                f"RMSE {self.train_rmse.result().numpy():.5f} | "
                f"Res {self.train_res.result().numpy():.5f} || "
                f"val MAE {self.val_mae.result().numpy():.5f} | "
                f"val RMSE {self.val_rmse.result().numpy():.5f} | "
                f"val Res {self.val_res.result().numpy():.5f}"
            )
