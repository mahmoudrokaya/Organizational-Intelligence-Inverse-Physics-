# src/data_loader.py
import numpy as np
import tensorflow as tf


def _to_py_path(p):
    """
    Convert tf.string / EagerTensor / bytes / str into a Python str path safely.
    """
    # p could be: tf.Tensor (EagerTensor), bytes, bytearray, or str
    if hasattr(p, "numpy"):  # tf.EagerTensor / tf.Tensor in eager mode
        p = p.numpy()
    if isinstance(p, bytearray):
        p = bytes(p)
    if isinstance(p, bytes):
        return p.decode("utf-8")
    if isinstance(p, str):
        return p
    # Last resort
    return str(p)


def _read_npz_py(path_in):
    path = _to_py_path(path_in)
    z = np.load(path, allow_pickle=False)

    # Stored as (T,H,W)
    u = z["u"].astype(np.float32)
    y = z["y"].astype(np.float32)

    # masks stored as (H,W) and (T,H,W) with 0/1
    m = z["sensor_mask"].astype(np.float32)      # kept for completeness (not used below)
    mt = z["sensor_mask_t"].astype(np.float32)

    # optional fields
    c = z["c_field"].astype(np.float32) if "c_field" in z else np.zeros((u.shape[1], u.shape[2]), np.float32)
    dt = np.float32(z["dt"]) if "dt" in z else np.float32(1.0)
    dx = np.float32(z["dx"]) if "dx" in z else np.float32(1.0)

    # build input channels: [y, mt] -> (T,H,W,2)
    x = np.stack([y, mt], axis=-1).astype(np.float32)

    # target: u -> (T,H,W,1)
    t = u[..., None].astype(np.float32)

    # IMPORTANT: return path as bytes for tf.string
    path_bytes = path.encode("utf-8")

    return x, t, c, dt, dx, path_bytes


def make_dataset(npz_files, batch_size=1, shuffle=True, repeat=False, deterministic=False):
    paths = tf.constant(npz_files, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(npz_files), 512), reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    ds = ds.with_options(options)

    def _read_tf(p):
        x, t, c, dt, dx, path = tf.py_function(
            func=_read_npz_py,
            inp=[p],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string],
        )

        # Shapes are unknown to TF after py_function; set them partially:
        x.set_shape([None, None, None, 2])   # (T,H,W,2)
        t.set_shape([None, None, None, 1])   # (T,H,W,1)
        c.set_shape([None, None])            # (H,W)
        dt.set_shape([])                     # scalar
        dx.set_shape([])                     # scalar
        path.set_shape([])                   # scalar string

        features = {"x": x, "c_field": c, "dt": dt, "dx": dx, "path": path}
        return features, t

    ds = ds.map(_read_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds
