"""
Kimi K3 from Scratch: KDA, Attention Residuals, and Stable LatentMoE

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - short_conv
import numpy as np

def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    # TODO: accumulate each kernel tap with a shifted slice add
    y = np.zeros_like(x)
    K = w.shape[0]
    T = x.shape[0]
    for j in range(K):
        lag = K - 1 - j
        if lag < T:
            y[lag:] += x[:T - lag] * w[j]

    return y

# Step 2 - kda_qkv
def kda_qkv(x, params):
    """KDA projections: q,k = L2Norm(Swish(ShortConv(W x))), v = Swish(ShortConv(Wv x)).

    params: dict with Wq (d,dk), Wk (d,dk), Wv (d,dv), cq (K,dk), ck (K,dk), cv (K,dv).
    Returns (q, k, v).  L2Norm divides each row by sqrt(sum(row**2) + 1e-6).
    """
    # TODO: project -> short_conv -> swish, then L2-normalize q and k rows
    q_proj = x @ params["Wq"] # (T, d) @ (d, dk) -> (T, dk)
    k_proj = x @ params["Wk"] # (T, dk)
    v_proj = x @ params["Wv"] # (T, dk)

    q_conv = short_conv(q_proj, params["cq"]) # (T, dk)
    k_conv = short_conv(k_proj, params["ck"]) # (T, dk)
    v_conv = short_conv(v_proj, params["cv"]) # (T, dk)

    q = swish(q_conv)
    k = swish(k_conv)
    v = swish(v_conv)

    q = l2_norm(q)
    k = l2_norm(k)

    return (q, k ,v)

def swish(z):
    return z / (1 + np.exp(-z))

def l2_norm(x):
    return x / np.sqrt(np.sum(x**2, axis=1, keepdims=True) + 1e-6)

# Step 3 - kda_gates
def kda_gates(x, params):
    """Return (beta, z): write strength sigmoid(x@wb+bb), decay logits x@Wd1@Wd2+ba.

    params: wb (d,), bb scalar, Wd1 (d,r), Wd2 (r,dk), ba (dk,).
    beta: (T,) in (0,1).  z: (T, dk), unbounded.
    """
    
    beta = sigmoid(x @ params["wb"] + params["bb"]) # (T,)
    z = x @ params["Wd1"] @ params["Wd2"] + params["ba"] # (T, dk)
    return (beta, z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 4 - lower_bounded_decay
def lower_bounded_decay(z, A, g_min=-5.0):
    """alpha = exp(g_min * sigmoid(exp(A) * z)), each entry in [exp(g_min), 1).

    z: (T, dk) decay logits.  A: scalar per-head log-scale.
    """
    g = g_min * sigmoid(np.exp(A) * z)
    alpha = np.exp(g)
    return alpha

# Step 5 - kda_state_update
def kda_state_update(S, k, v, alpha, beta):
    """One KDA step: (I - beta k k^T) @ diag(alpha) @ S + beta * outer(k, v).

    S: (dk, dv).  k: (dk,).  v: (dv,).  alpha: (dk,).  beta: scalar.
    """
    # TODO: decay, then delta-rule erase, then write
    S = alpha[:, None] * S                  # decay: diag(alpha) @ S，按行缩放
    S = S - beta * np.outer(k, k @ S)       # erase: (I - beta k k^T) @ S
    S = S + beta * np.outer(k, v)           # write
    return S

# Step 6 - kda_recurrence
def kda_recurrence(q, k, v, alpha, beta, S0=None):
    """Run KDA token by token: update state, then read O[t] = S_t^T q[t].

    Returns (O, S_final) with O of shape (T, dv). S0 defaults to zeros; never
    mutate the caller's S0.
    """
    T = k.shape[0]
    dk = k.shape[1]
    dv = v.shape[1]
    S = np.zeros((dk, dv)) if S0 is None else np.array(S0, dtype=float)
    O = np.empty((T, dv), dtype=float)
    for t in range(T):
        S = kda_state_update(S, k[t], v[t], alpha[t], beta[t])
        O[t] = S.T @ q[t]

    return (O, S)

# Step 7 - cumulative_decay
def cumulative_decay(alpha):
    """Inclusive channel-wise cumulative product of alpha down the time axis.

    alpha: (C, dk) per-step retention factors -> Gamma: (C, dk).
    """
    T = alpha.shape[0]
    Gamma = np.empty_like(alpha)
    Gamma[0] = alpha[0]
    for i in range(1, T):
        Gamma[i] = Gamma[i-1] * alpha[i]

    return Gamma
    # return np.cumprod(alpha, axis=0)

# Step 8 - chunk_pseudo_values
def chunk_pseudo_values(k, v, alpha, beta, S0):
    """Solve (I + diag(beta) strict_tril(Khat Kcheck^T)) U = diag(beta)(V - Khat S0).

    Khat = k * Gamma, Kcheck = k / Gamma, Gamma = cumulative_decay(alpha).
    Returns U of shape (C, dv).
    """
    Gamma = cumulative_decay(alpha) # (C, dk)
    Khat = k * Gamma                # (C, dk)
    Kcheck = k / Gamma              # (C, dk)

    L = beta[:, None] * np.tril(Khat @ Kcheck.T, k = -1) # (C, C)，严格下三角
    rhs = beta[:, None] * (v - Khat @ S0) # (C, dv)

    C = k.shape[0]
    U = np.empty_like(rhs)  # (C, dv)
    for i in range(C):
        U[i] = rhs[i] - L[i, :i] @ U[:i]

    return U

# Step 9 - kda_chunkwise
def kda_chunkwise(q, k, v, alpha, beta, chunk_size, S0=None):
    """Chunkwise-parallel KDA (Eq. 4): O_c = Qhat @ S + tril(Qhat Kcheck^T) @ U.

    State hand-off: S <- Gamma[-1][:,None] * (S + Kcheck^T U). Must equal
    kda_recurrence for every chunk size. Returns (O, S_final).
    """
    T, dk = k.shape
    dv = v.shape[1]
    S = np.zeros((dk, dv)) if S0 is None else np.array(S0, dtype=float)
    O = np.empty((T, dv))

    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)                       # 最后一块可能不满
        q_c, k_c, v_c = q[s:e], k[s:e], v[s:e]           # (C, dk) / (C, dv)
        alpha_c, beta_c = alpha[s:e], beta[s:e]          # (C, dk) / (C,)

        Gamma = cumulative_decay(alpha_c)                # (C, dk)
        Qhat = q_c * Gamma                               # (C, dk)
        Kcheck = k_c / Gamma                             # (C, dk)
        U = chunk_pseudo_values(k_c ,v_c, alpha_c, beta_c, S) # (C, dv)
        O_new = np.tril(Qhat @ Kcheck.T, k = 0) @ U      # (C, dv)
        O_history = Qhat @ S                             # (C, dv)
        O[s:e] = O_history + O_new

        S = Gamma[-1][:, None] * (S + Kcheck.T @ U)      # (dk, dv)

    return (O, S)

# Step 10 - kda_output_gate
def kda_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo, RMSNorm = o / sqrt(mean(o^2)+1e-6).

    o: (T, dv) recurrent outputs.  x: (T, d) layer input.  Returns (T, d).
    """
    y = (sigmoid(x @ Wg) * RMSNorm(o)) @ Wo
    return y


def RMSNorm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) # (T, d) / (T,1)

# Step 11 - mla_compress_reconstruct
def mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads):
    """c = x @ Wc; K = (c @ Wk_up).reshape(T, H, dh); V likewise.

    Returns (c, K, V) with shapes (T, r), (T, H, dh), (T, H, dh).
    """
    T = x.shape[0]
    H = n_heads

    c = x @ Wc                   # (T, r)
    K = (c @ Wk_up).reshape(T, H, -1) # (T, H, dh)
    V = (c @ Wv_up).reshape(T, H, -1) # (T, H, dh)

    return (c, K, V)

# Step 12 - nope_attention
def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T = x.shape[0]
    H = n_heads

    Q = (x @ Wq).reshape(T, H, -1) # (T, H, dh)
    _, K, V = mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, H) # K/V: (T, H, dh)
    dh = Q.shape[-1]

    scale = 1.0 / np.sqrt(dh)
    qk = np.einsum('thd,shd->hts', Q * scale, K)   # (H, T, T)

    mask = np.tril(np.ones((T, T), dtype=bool), k=0) # causal mask
    qk = np.where(mask, qk, -np.inf)
    P = softmax(qk)                        # (H, T, T)
    O = np.einsum('hts,shd->thd', P, V)    # (T, H, dh)

    return O.reshape(T, -1)                # (T, H*dh)


def softmax(x, axis=-1):
    row_max = np.max(x, axis=axis, keepdims=True)  # (H, T, 1)
    x = x - row_max                              # (H, T, T)
    P = np.exp(x)                                # (H, T, T)
    row_sum = np.sum(P, axis=axis, keepdims=True)  # (H, T, 1)
    return P / row_sum                           # (H, T, T)

# Step 13 - mla_output_gate
def mla_output_gate(o, x, Wg, Wo):
    """y = (sigmoid(x @ Wg) * o) @ Wo - note: no RMSNorm here, unlike KDA's gate.

    o: (T, H*dh) attention output.  x: (T, d) layer input.  Returns (T, d).
    """
    return (sigmoid(x @ Wg) * o) @ Wo

# Step 14 - hybrid_schedule
def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    return ['KDA','KDA','KDA','MLA'] * n_repeats + ['MLA']

# Step 15 - attnres_weights
def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    V = np.stack(sources)       # (n, T, d)
    Vn = RMSNorm(V)             # (n, T, d)
    s = Vn @ pseudo_q           # (n, T)
    w = softmax(s, axis=0)      # (n, T)

    return w

# Step 16 - attnres_full
def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    w = attnres_weights(pseudo_q, sources) # (n, T)
    V = np.stack(sources)                  # (n, T, d)
    h = np.einsum('nt,ntd->td', w, V)      # (T, d)
    return h

# Step 17 - block_partial_sums
def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    partial = []
    acc = np.zeros_like(layer_outputs[0], dtype=float)   # (T, d)
    for out in layer_outputs:
        acc = acc + out          # 生成新数组，不是原地修改
        partial.append(acc)

    return partial

# Step 18 - attnres_block
def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).

    partial is None for the first layer of a block. Returns (T, d).
    """
    sources = list(block_reps)          # 拷贝 list，不修改调用方的 block_reps
    if partial is not None:
        sources.append(partial)         # 当前 block 内已有层输出的累加和
    return attnres_full(pseudo_q, sources) # (T, d)

# Step 19 - situ_glu
def situ_glu(x, Wg, Wu, beta1=4.0, beta2=25.0):
    """(softcap(x@Wg, b1) * sigmoid(x@Wg)) * softcap(x@Wu, b2), softcap = b*tanh(u/b).

    Use sigmoid(g) = 0.5*(1 + tanh(0.5*g)) to stay overflow-safe. |out| <= b1*b2.
    """
    g = x @ Wg  # (T, d) @ (d, d_ff) -> (T, d_ff)
    u = x @ Wu  # (T, d_ff)
    inv_beta1 = 1.0 / beta1 # 标量
    inv_beta2 = 1.0 / beta2 # 标量

    sigmoid_g = 0.5 * (1.0 + np.tanh(0.5 * g)) # (T, d_ff)
    gate = beta1 * np.tanh(g * inv_beta1) * sigmoid_g # (T, d_ff), softplus(x, b) = b * tanh(x / b)
    up = beta2 * np.tanh(u * inv_beta2) # (T, d_ff)
    return gate * up

# Step 20 - route_topk
def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    s = sigmoid(x @ Wr) # (T, d) @ (d, n) -> (T, n)
    idx = np.argsort(-(s + bias), axis=1, kind='stable')[:, :k] # (T, k), 每一行得到 tok-K index
    selected_score = np.take_along_axis(s, idx, axis=1)             # (T, k), 每一行取出 tok-K 分数
    p = selected_score / selected_score.sum(axis=1, keepdims=True)  # (T, k)

    return (s, idx, p)

# Step 21 - routed_experts
def routed_experts(z, idx, p, experts):
    """u[t] = sum_j p[t,j] * situ_glu(z[t], *experts[idx[t,j]]).

    z: (T, l).  experts: list of (Wg, Wu) latent-width SiTU-GLUs. Returns (T, l).
    """
    u = np.zeros_like(z)

    for expert_id, (Wg, Wu) in enumerate(experts): # 遍历每个 expert 权重
        token_idx, slot_idx = np.nonzero(idx == expert_id)
        if token_idx.size == 0: # 没有 token 路由到
            continue
        
        tokens = z[token_idx]  # (M, l), 从 z 中拷贝出选中的 token latent
        outputs = situ_glu(tokens, Wg, Wu)        # (M, l)
        weights = p[token_idx, slot_idx][:, None] # (M, 1)

        np.add.at(u, token_idx, outputs * weights) # scatter-add

    return u

# Step 22 - stable_latent_moe
def stable_latent_moe(x, params):
    """y = sum_shared SiTU(x) + RMSNorm(routed_aggregate) @ Wup (Eq. 11).

    Route on full-width x; compute in latent width; RMSNorm(u) before Wup.
    """
    z = x @ params["Wdown"] # (T, d) @ (d, l) -> (T, l)
    s, idx, p = route_topk(x, params["Wr"], params["bias"], params["k"])
    u = routed_experts(z, idx, p, params["experts"]) # (T, l)
    normed_u = RMSNorm(u) # (T, l)

    y = normed_u @ params["Wup"] # (T, d)

    for Wg_s, Wu_s in params["shared"]:
        y += situ_glu(x, Wg_s, Wu_s) # (T, d) @ (d, d) -> (T, d)

    return y

# Step 23 - topk_cutoffs
def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    biased_scores = s + bias # (m, n)
    order = np.argsort(-biased_scores, axis=1, kind="stable") # 每一行降序排列，返回原始位置 index

    routes = order[:, :k] # 前 k 个专家编号，(m, k)
    m = s.shape[0]
    cutoffs = biased_scores[np.arange(m), order[:, k]]  # 第 k+1 个专家的 biased score，(m,)

    return (routes, cutoffs)

# Step 24 - quantile_balance_update
def quantile_balance_update(s, bias, k):
    """QB (Eq. 14): bhat_j = -(the (q+1)-th largest of s[:, j] - cutoffs),
    q = m*k // n; return bhat - mean(bhat).
    """
    m, n = s.shape
    _, cutoffs = topk_cutoffs(s, bias, k) # cutoffs: (m,)
    margins = s - cutoffs[:, None]        # (m, n)
    q = (m*k) // n
    bhat = np.sort(-margins, axis=0)[q, :] # (n,), 每一列取 (q+1) 小
    return bhat - bhat.mean()

# Step 25 - histogram_quantile
def histogram_quantile(x, n_bins, lo, hi, q_frac):
    """Quantile from pooled bin counts; error <= (hi - lo) / n_bins.

    Return the right edge of the first bin whose cumulative count reaches
    q_frac * len(x).
    """
    counts, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    # counts  (n_bins,)，每个桶的样本数量
    # edges   (n_bins + 1,)，所有桶边界
    cumulative = np.cumsum(counts) # (n_bins,)，累计数量

    # 找到第一个满足 cumulative[bin_idx] >= q_frac * len(x) 的位置
    bin_idx = np.searchsorted(cumulative, q_frac * len(x), side="left")
    bin_idx = min(bin_idx, n_bins - 1) 

    return edges[bin_idx + 1] # 返回右边界

# Step 26 - newton_schulz
def newton_schulz(G, n_iters=5):
    """Muon's Newton-Schulz orthogonalization (a,b,c = 3.4445, -4.7750, 2.0315).

    Normalize by the Frobenius norm (+1e-7), iterate the quintic, transpose
    handling for tall matrices. Singular values -> 1.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    transposed = G.shape[0] > G.shape[1]
    X = G.T if transposed else G  # 让 G 的行数小于列数，从而 X @ X^T 更小

    X = X / (np.linalg.norm(X) + 1e-7) # Frobenius norm

    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.T if transposed else X

# Step 27 - per_head_muon
def per_head_muon(M, n_heads, n_iters=5):
    """Split M (d, H*dh) into H column blocks, newton_schulz each, re-concatenate."""
    dh = M.shape[1] // n_heads
    M_out = np.empty_like(M)

    for h in range(n_heads):
        start = h * dh
        end = start + dh

        block = M[:, start:end]  # (d, dh)
        M_out[:, start:end] = newton_schulz(block, n_iters)

    return M_out

# Step 28 - mini_k3_forward
def mini_k3_forward(tokens, seed):
     """Miniature Kimi K3 forward pass. Returns (T, 17) logits.

     Parameter creation order (all rng.normal(0.0, 0.3, size=...)):
      emb (17, 16)
      then for each layer of hybrid_schedule(1):
        wq_res (16,)
        KDA: Wq (16,8), Wk (16,8), Wv (16,8), cq (3,8), ck (3,8), cv (3,8),
             wb (16,), Wd1 (16,4), Wd2 (4,8), ba (8,), Wg (16,8), Wo (8,16)
             (bb = 0.0 and A = 0.0 are constants, not drawn)
        MLA: Wq (16,16), Wc (16,8), Wk_up (8,16), Wv_up (8,16),
             Wg (16,16), Wo (16,16)
        MoE: Wdown (16,8), Wup (8,16), Wr (16,8), bias = zeros(8), k=2,
             experts: 8 x (Wg (8,8), Wu (8,8)) drawn gate-then-up per expert,
             shared: 2 x (Wg (16,16), Wu (16,16)) drawn gate-then-up
      w_final (16,)
     """
     rng = np.random.default_rng(seed)
     def W(*shape):
          return rng.normal(0.0, 0.3, size=shape)

     # 模型与词表
     vocab_size = 17
     d_model = 16

     # KDA
     dk = 8
     dv = 8
     conv_kernel_size = 3
     decay_rank = 4

     # MLA
     n_heads = 2
     dh = 8
     d_kv_latent = 8

     # MoE
     d_moe_latent = 8
     n_experts = 8
     top_k = 2
     n_shared_experts = 2
     
     embedding = W(vocab_size, d_model)
     layers = []

     for kind in hybrid_schedule(1):
          wq_res = W(d_model)

          if kind == "KDA":
               attn = {
                    "Wq": W(d_model, dk),
                    "Wk": W(d_model, dk),
                    "Wv": W(d_model, dv),
                    "cq": W(conv_kernel_size, dk),
                    "ck": W(conv_kernel_size, dk),
                    "cv": W(conv_kernel_size, dv),
                    "wb": W(d_model),
                    "Wd1": W(d_model, decay_rank),
                    "Wd2": W(decay_rank, dk),
                    "ba": W(dk),
                    "Wg": W(d_model, dv),
                    "Wo": W(dv, d_model),
                    "bb": 0.0,
               }
          else: # MLA
               attn = {
                    "Wq": W(d_model, n_heads * dh),
                    "Wc": W(d_model, d_kv_latent),
                    "Wk_up": W(d_kv_latent, n_heads * dh),
                    "Wv_up": W(d_kv_latent, n_heads * dh),
                    "Wg": W(d_model, n_heads * dh),
                    "Wo": W(n_heads * dh, d_model),
               }

          moe = {
               "Wdown": W(d_model, d_moe_latent),
               "Wup": W(d_moe_latent, d_model),
               "Wr": W(d_model, n_experts),
               "bias": np.zeros(n_experts),
               "k": 2,
               "experts": [
                    (
                         W(d_moe_latent, d_moe_latent),
                         W(d_moe_latent, d_moe_latent),
                    )
                    for _ in range(n_experts)
               ],
               "shared": [
                    (
                         W(d_model, d_model),
                         W(d_model, d_model),
                    )
                    for _ in range(n_shared_experts)
               ],
          }

          layers.append((kind, wq_res, attn, moe))

     w_final = W(d_model)

     # 执行各层 froward
     sources = [embedding[np.asarray(tokens, dtype=np.int64)]]  # (T, 16)

     for kind, wq_res, attn, moe in layers:
          h = attnres_full(wq_res, sources)  # (T, 16)

          if kind == "KDA":
               q, k, v = kda_qkv(h, attn)
               beta, decay_logits = kda_gates(h, attn)
               alpha = lower_bounded_decay(decay_logits, A=0.0)

               # o, _ = kda_recurrence(q, k, v, alpha, beta)
               o, _ = kda_chunkwise(q, k, v, alpha, beta, chunk_size=4)
               a = kda_output_gate(o, h, attn["Wg"], attn["Wo"])
          else:
               o = nope_attention(
                    h,
                    attn["Wq"],
                    attn["Wc"],
                    attn["Wk_up"],
                    attn["Wv_up"],
                    n_heads=2,
               )
               a = mla_output_gate(o, h, attn["Wg"], attn["Wo"])

          f = a + stable_latent_moe(a, moe)  # (T, 16)
          sources.append(f)

     # 最终的 depth read 与词表投影
     h_out = attnres_full(w_final, sources)  # (T, 16)
     return h_out @ embedding.T              # (T, 17)

