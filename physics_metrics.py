import tensorflow as tf

@tf.function
def laplacian_2d(u_hw, dx):
    # u_hw: (H,W)
    u = u_hw
    p = tf.pad(u, [[1,1],[1,1]], mode="REFLECT")
    lap = (p[:-2,1:-1] + p[2:,1:-1] + p[1:-1,:-2] + p[1:-1,2:] - 4.0*p[1:-1,1:-1]) / (dx*dx)
    return lap

@tf.function
def wave_residual_norm(u_hat, c_field, dt, dx):
    """
    u_hat: (B,T,H,W,1)
    c_field: (B,H,W) or (H,W)
    returns: scalar residual norm (mean L2 over time & space)
    Residual: u[t+1]-2u[t]+u[t-1] - dt^2 * c^2 * Lap(u[t])
    """
    u_hat = tf.squeeze(u_hat, axis=-1)  # (B,T,H,W)
    B = tf.shape(u_hat)[0]
    T = tf.shape(u_hat)[1]

    # ensure c_field has batch dim
    if len(c_field.shape) == 2:
        c = tf.broadcast_to(c_field[None, ...], [B, tf.shape(c_field)[0], tf.shape(c_field)[1]])
    else:
        c = c_field

    dt2 = dt * dt
    # compute residual for t=1..T-2
    u_prev = u_hat[:, :-2, :, :]
    u_mid  = u_hat[:, 1:-1, :, :]
    u_next = u_hat[:, 2:, :, :]

    # laplacian per (B, T-2) slice
    def lap_one(b_t):
        b, t = b_t[0], b_t[1]
        return laplacian_2d(u_mid[b, t], dx)

    # vectorized map
    BT = tf.shape(u_mid)[0] * tf.shape(u_mid)[1]
    b_idx = tf.repeat(tf.range(B), repeats=(T-2))
    t_idx = tf.tile(tf.range(T-2), multiples=[B])
    bt = tf.stack([b_idx, t_idx], axis=1)
    laps = tf.map_fn(lap_one, bt, fn_output_signature=tf.float32)
    laps = tf.reshape(laps, [B, T-2, tf.shape(u_mid)[2], tf.shape(u_mid)[3]])

    c2 = tf.expand_dims(c*c, axis=1)  # (B,1,H,W)
    r = (u_next - 2.0*u_mid + u_prev) - dt2 * c2 * laps
    # mean L2
    r2 = tf.reduce_mean(tf.square(r))
    return tf.sqrt(r2 + 1e-12)
