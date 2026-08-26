
import tensorflow as tf

EPS = tf.constant(1e-8, tf.float32)

@tf.function
def sensor_consistency_score(y_patch, x_patch):
    y_obs = tf.cast(x_patch[..., 0:1], tf.float32)
    mask = tf.cast(x_patch[..., 1:2], tf.float32)
    axes = [1,2,3,4]
    nobs = tf.reduce_sum(mask, axis=axes)
    err = tf.reduce_sum(tf.abs(tf.cast(y_patch, tf.float32)-y_obs)*mask, axis=axes)
    score = tf.where(nobs > 0, err/tf.maximum(nobs, 1.0), tf.zeros_like(nobs))
    total = tf.cast(tf.reduce_prod(tf.shape(mask)[1:]), tf.float32)
    coverage = nobs/tf.maximum(total, 1.0)
    return score, coverage

@tf.function
def wave_residual_score_per_sample(y_patch, c_patch, dt, dx):
    u = tf.squeeze(tf.cast(y_patch, tf.float32), -1)
    B, H, W = tf.shape(u)[0], tf.shape(u)[2], tf.shape(u)[3]

    c_patch = tf.cast(c_patch, tf.float32)
    if c_patch.shape.rank == 2:
        c = tf.broadcast_to(c_patch[None, ...], [B,H,W])
    else:
        c = c_patch

    dt = tf.reshape(tf.cast(dt, tf.float32), [-1])[0]
    dx = tf.reshape(tf.cast(dx, tf.float32), [-1])[0]

    um = u[:,1:-1,:,:]
    BT = tf.shape(um)[0]*tf.shape(um)[1]
    u2 = tf.reshape(um, [BT,H,W,1])

    k = tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], tf.float32)
    k = k[:,:,None,None]
    u2p = tf.pad(u2, [[0,0],[1,1],[1,1],[0,0]], mode="REFLECT")
    lap = tf.nn.conv2d(u2p, k, 1, "VALID")
    lap = tf.reshape(lap[...,0], [B,tf.shape(um)[1],H,W])/(dx*dx+EPS)

    r = (u[:,2:,:,:] - 2.0*um + u[:,:-2,:,:]) - (dt*dt)*tf.expand_dims(c*c,1)*lap
    return tf.sqrt(tf.reduce_mean(tf.square(r), axis=[1,2,3]) + EPS)

@tf.function
def normalized_gate_entropy(g):
    g = tf.clip_by_value(tf.cast(g, tf.float32), EPS, 1.0)
    h = -tf.reduce_sum(g*tf.math.log(g), axis=1)
    K = tf.cast(tf.shape(g)[1], tf.float32)
    return h/tf.math.log(tf.maximum(K, 2.0))

@tf.function
def normalize_agents(s):
    smin = tf.reduce_min(s, axis=1, keepdims=True)
    smax = tf.reduce_max(s, axis=1, keepdims=True)
    return (s-smin)/(smax-smin+EPS)

def compute_deployment_influence_weights(
    patch_outs, gates, regions, x, c_field, dt, dx,
    sensor_weight=0.50, physics_weight=0.35,
    entropy_weight=0.15, temperature=5.0
):
    # IMPORTANT: there is intentionally NO y_true argument.
    ss, ps, es, covs = [], [], [], []

    for i,(y0,y1,x0,x1) in enumerate(regions):
        xp = x[:,:,y0:y1,x0:x1,:]
        cp = c_field[y0:y1,x0:x1] if c_field.shape.rank == 2 else c_field[:,y0:y1,x0:x1]

        s,cov = sensor_consistency_score(patch_outs[i], xp)
        p = wave_residual_score_per_sample(patch_outs[i], cp, dt, dx)
        e = normalized_gate_entropy(gates[i])

        ss.append(s); ps.append(p); es.append(e); covs.append(cov)

    S = tf.stack(ss,1); P = tf.stack(ps,1); E = tf.stack(es,1); C = tf.stack(covs,1)
    Sn, Pn, En = normalize_agents(S), normalize_agents(P), normalize_agents(E)

    available = tf.cast(C > 0.0, tf.float32)
    ws, wp, we = map(lambda z: tf.cast(z, tf.float32),
                     [sensor_weight, physics_weight, entropy_weight])

    scale = ws*available + wp + we
    composite = (ws*Sn*available + wp*Pn + we*En)/tf.maximum(scale, EPS)
    weights = tf.nn.softmax(-tf.cast(temperature, tf.float32)*composite, axis=1)

    return weights, {
        "sensor_score": S,
        "physics_score": P,
        "gate_entropy": E,
        "sensor_coverage": C,
        "composite_score": composite,
    }
