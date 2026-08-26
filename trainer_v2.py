
import tensorflow as tf
from src.physics_metrics import wave_residual_norm
from src.models_sacu import stitch_patches
from src.inference_weights import compute_deployment_influence_weights

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred))

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)) + 1e-12)

def predict_sacu_deployment(
    model, x, c_field, dt, dx, training=False,
    sensor_weight=0.50, physics_weight=0.35,
    entropy_weight=0.15, temperature=5.0
):
    # IMPORTANT: no y_true parameter.
    _, aux = model(x, training=training)
    patch_outs, regions, gates = aux["patch_outs"], aux["regions"], aux["gates"]

    weights, diag = compute_deployment_influence_weights(
        patch_outs, gates, regions, x, c_field, dt, dx,
        sensor_weight, physics_weight, entropy_weight, temperature
    )

    y_pred = stitch_patches(
        patch_outs, regions, weights, tf.shape(x)[2], tf.shape(x)[3]
    )
    diag["influence_weights"] = weights
    diag["gates"] = tf.stack(gates, axis=1)
    return y_pred, diag

class TrainerV2:
    def __init__(
        self, model, optimizer, use_physics_loss=True, lambda_phys=0.05,
        sensor_weight=0.50, physics_weight=0.35,
        entropy_weight=0.15, temperature=5.0
    ):
        self.model = model
        self.opt = optimizer
        self.use_physics_loss = use_physics_loss
        self.lambda_phys = lambda_phys
        self.sensor_weight = sensor_weight
        self.physics_weight = physics_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature

    def _predict(self, x, c, dt, dx, training):
        return predict_sacu_deployment(
            self.model, x, c, dt, dx, training,
            self.sensor_weight, self.physics_weight,
            self.entropy_weight, self.temperature
        )

    @tf.function(reduce_retracing=True)
    def train_step_sacu(self, x, y_true, c, dt, dx):
        with tf.GradientTape() as tape:
            # Prediction is fully constructed BEFORE y_true is used.
            y_pred, diag = self._predict(x, c, dt, dx, True)
            data_loss = tf.reduce_mean(tf.square(y_pred-y_true))
            residual = wave_residual_norm(y_pred, c, dt, dx)
            loss = data_loss + (self.lambda_phys*residual if self.use_physics_loss else 0.0)

        grads = tape.gradient(loss, self.model.trainable_variables)
        gv = [(g,v) for g,v in zip(grads,self.model.trainable_variables) if g is not None]
        self.opt.apply_gradients(gv)

        return loss, mae(y_true,y_pred), rmse(y_true,y_pred), residual

    @tf.function(reduce_retracing=True)
    def eval_step_sacu(self, x, y_true, c, dt, dx):
        # y_true is used only for metrics after prediction exists.
        y_pred, diag = self._predict(x, c, dt, dx, False)
        return (
            mae(y_true,y_pred),
            rmse(y_true,y_pred),
            wave_residual_norm(y_pred,c,dt,dx),
            diag["influence_weights"]
        )
