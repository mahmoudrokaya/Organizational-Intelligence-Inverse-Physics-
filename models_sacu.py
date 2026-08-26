import tensorflow as tf
from tensorflow.keras import layers

# TF-compatible serialization decorator
register_keras_serializable = tf.keras.utils.register_keras_serializable


@register_keras_serializable(package="gis")
class MicroExpert(layers.Layer):
    """Two-layer 1x1x1 Conv => voxel-wise shallow mapping."""
    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden = int(hidden)
        self.c1 = layers.Conv3D(self.hidden, 1, padding="same", activation="relu", name="c1")
        self.c2 = layers.Conv3D(1, 1, padding="same", activation=None, name="c2")

    def build(self, input_shape):
        # input_shape: (B,T,h,w,C)
        self.c1.build(input_shape)
        # output channels become self.hidden
        self.c2.build((None, input_shape[1], input_shape[2], input_shape[3], self.hidden))
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.c1(x, training=training)
        x = self.c2(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden": self.hidden})
        return cfg


@register_keras_serializable(package="gis")
class SACU(layers.Layer):
    """
    Shallow Adaptive Cooperative Unit:
      - K micro-experts
      - gating uses pooled features + optional role embedding + optional message vector
    Important: gate input dimension MUST be constant across calls.
    """
    def __init__(
        self,
        K=4,
        hidden=64,
        msg_dim=16,
        role_vocab=8,
        use_role=True,
        use_comms=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.K = int(K)
        self.hidden = int(hidden)
        self.msg_dim = int(msg_dim)
        self.role_vocab = int(role_vocab)
        self.use_role = bool(use_role)
        self.use_comms = bool(use_comms)

        # Experts registered as attributes (robust saving/loading)
        self.experts = []
        for i in range(self.K):
            e = MicroExpert(hidden=self.hidden, name=f"expert_{i}")
            self.experts.append(e)
            setattr(self, f"expert_{i}", e)

        self.pool = layers.GlobalAveragePooling3D(name="pool")

        self.role_emb = layers.Embedding(self.role_vocab, 16, name="role_emb") if self.use_role else None
        self.msg_proj = layers.Dense(16, activation="relu", name="msg_proj") if self.use_comms else None

        self.gate = layers.Dense(self.K, activation="softmax", name="gate")
        self.msg_out = layers.Dense(self.msg_dim, activation=None, name="msg_out")

    def build(self, input_shape):
        # input_shape: (B,T,h,w,C)
        for e in self.experts:
            e.build(input_shape)

        C = int(input_shape[-1])
        ctx_dim = C
        if self.use_role:
            ctx_dim += 16
        if self.use_comms:
            self.msg_proj.build((None, self.msg_dim))
            ctx_dim += 16

        self.gate.build((None, ctx_dim))
        self.msg_out.build((None, ctx_dim))
        super().build(input_shape)

    def call(self, x_patch, role_id=None, msg_in=None, training=False):
        """
        Always keep ctx dimension consistent with build():
        - if use_role and role_id is None -> use zeros role_id
        - if use_comms and msg_in is None -> use zeros msg_in
        """
        ctx = self.pool(x_patch)  # (B,C)
        B = tf.shape(ctx)[0]

        if self.use_role:
            if role_id is None:
                role_id = tf.zeros([B], dtype=tf.int32)
            r = self.role_emb(role_id)  # (B,16)
            ctx = tf.concat([ctx, r], axis=-1)

        if self.use_comms:
            if msg_in is None:
                msg_in = tf.zeros([B, self.msg_dim], dtype=ctx.dtype)
            m = self.msg_proj(msg_in)  # (B,16)
            ctx = tf.concat([ctx, m], axis=-1)

        g = self.gate(ctx)  # (B,K)

        outs = [e(x_patch, training=training) for e in self.experts]
        outs = tf.stack(outs, axis=1)  # (B,K,T,h,w,1)

        g_ = tf.reshape(g, [-1, self.K, 1, 1, 1, 1])
        y = tf.reduce_sum(g_ * outs, axis=1)  # (B,T,h,w,1)

        m_out = self.msg_out(ctx)  # (B,msg_dim)
        return y, g, m_out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "K": self.K,
            "hidden": self.hidden,
            "msg_dim": self.msg_dim,
            "role_vocab": self.role_vocab,
            "use_role": self.use_role,
            "use_comms": self.use_comms,
        })
        return cfg


def make_regions(H, W, grid=4, overlap=8):
    regions = []
    gh = grid
    gw = grid
    h0s = [int(round(i * H / gh)) for i in range(gh)]
    h1s = [int(round((i + 1) * H / gh)) for i in range(gh)]
    w0s = [int(round(j * W / gw)) for j in range(gw)]
    w1s = [int(round((j + 1) * W / gw)) for j in range(gw)]

    for i in range(gh):
        for j in range(gw):
            y0, y1 = h0s[i], h1s[i]
            x0, x1 = w0s[j], w1s[j]
            y0o = max(0, y0 - overlap)
            x0o = max(0, x0 - overlap)
            y1o = min(H, y1 + overlap)
            x1o = min(W, x1 + overlap)
            regions.append((y0o, y1o, x0o, x1o))
    return regions


def neighbor_graph(grid=4):
    N = grid * grid
    neigh = [[] for _ in range(N)]
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < grid and 0 <= jj < grid:
                    neigh[idx].append(ii * grid + jj)
    return neigh


@register_keras_serializable(package="gis")
class OrgSACUSolver(tf.keras.Model):
    def __init__(
        self,
        grid=4,
        overlap=8,
        K=4,
        hidden=64,
        msg_dim=16,
        use_role=True,
        use_comms=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grid = int(grid)
        self.overlap = int(overlap)
        self.N = self.grid * self.grid
        self.K = int(K)
        self.hidden = int(hidden)
        self.msg_dim = int(msg_dim)
        self.use_role = bool(use_role)
        self.use_comms = bool(use_comms)

        self.sacus = []
        for i in range(self.N):
            s = SACU(
                K=self.K,
                hidden=self.hidden,
                msg_dim=self.msg_dim,
                use_role=self.use_role,
                use_comms=self.use_comms,
                name=f"sacu_{i}",
            )
            self.sacus.append(s)
            setattr(self, f"sacu_{i}", s)

        self.neigh = neighbor_graph(grid=self.grid)

    def build(self, input_shape):
        # input_shape: (B,T,H,W,C)
        T = int(input_shape[1])
        H = int(input_shape[2])
        W = int(input_shape[3])
        C = int(input_shape[4])

        regions = make_regions(H, W, grid=self.grid, overlap=self.overlap)
        for i, (y0, y1, x0, x1) in enumerate(regions):
            h = y1 - y0
            w = x1 - x0
            self.sacus[i].build((None, T, h, w, C))

        super().build(input_shape)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        Hs = int(x.shape[2])
        Ws = int(x.shape[3])
        regions = make_regions(Hs, Ws, grid=self.grid, overlap=self.overlap)

        role_ids = tf.range(self.N) % 8
        role_ids = tf.broadcast_to(role_ids[None, :], [B, self.N])  # (B,N)

        # IMPORTANT: if comms enabled, provide zero msg_in in the first pass too
        zero_msg = tf.zeros([B, self.msg_dim], dtype=x.dtype)

        msgs, gates, patch_outs = [], [], []

        # First pass
        for i, (y0, y1, x0, x1) in enumerate(regions):
            xp = x[:, :, y0:y1, x0:x1, :]
            rid = role_ids[:, i] if self.use_role else None
            m_in = zero_msg if self.use_comms else None
            y_hat, g, m_out = self.sacus[i](xp, role_id=rid, msg_in=m_in, training=training)
            patch_outs.append(y_hat)
            gates.append(g)
            msgs.append(m_out)

        msgs = tf.stack(msgs, axis=1)  # (B,N,msg_dim)

        # Optional comms round
        if self.use_comms:
            patch_outs2, gates2 = [], []
            for i, (y0, y1, x0, x1) in enumerate(regions):
                xp = x[:, :, y0:y1, x0:x1, :]
                rid = role_ids[:, i] if self.use_role else None
                nbs = self.neigh[i]
                if len(nbs) == 0:
                    m_in = zero_msg
                else:
                    m_in = tf.reduce_mean(tf.gather(msgs, nbs, axis=1), axis=1)
                y_hat, g, _ = self.sacus[i](xp, role_id=rid, msg_in=m_in, training=training)
                patch_outs2.append(y_hat)
                gates2.append(g)
            patch_outs, gates = patch_outs2, gates2

        aux = {"patch_outs": patch_outs, "regions": regions, "gates": gates}
        u0 = tf.zeros([B, T, Hs, Ws, 1], dtype=tf.float32)
        return u0, aux

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "grid": self.grid,
            "overlap": self.overlap,
            "K": self.K,
            "hidden": self.hidden,
            "msg_dim": self.msg_dim,
            "use_role": self.use_role,
            "use_comms": self.use_comms,
        })
        return cfg


def stitch_patches(patch_outs, regions, weights, H, W):
    B = tf.shape(patch_outs[0])[0]
    T = tf.shape(patch_outs[0])[1]

    num = tf.zeros([B, T, H, W, 1], tf.float32)
    den = tf.zeros([B, T, H, W, 1], tf.float32)

    for i, (y0, y1, x0, x1) in enumerate(regions):
        p = patch_outs[i]
        wi = tf.reshape(weights[:, i], [B, 1, 1, 1, 1])
        pw = p * wi

        pad = [[0, 0], [0, 0], [y0, H - y1], [x0, W - x1], [0, 0]]
        pw_full = tf.pad(pw, pad)
        m_full = tf.pad(tf.ones_like(pw), pad) * wi

        num += pw_full
        den += m_full

    return num / (den + 1e-6)