"""
Modified from: https://github.com/jaymody/picoGPT/blob/main/gpt2.py.
"""
import numpy as np

# control print truncation for readability
np.set_printoptions(edgeitems=3, threshold=10, linewidth=200)

def print_indent(tensor):
    """Helper to print a numpy tensor with indentation."""
    s = np.array2string(tensor, max_line_width=200)
    for line in s.splitlines():
        print("      " + line)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(variance + eps)
    return g * out + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    o = linear(a, **c_proj)
    return a, o


def attention(q, k, v, mask, idx, debug=False):
    if not debug or idx != 0:
        return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
    # debug
    print("    attention scores, shape =", (q @ k.T / np.sqrt(q.shape[-1])).shape)
    scores = q @ k.T / np.sqrt(q.shape[-1]) + mask
    print("      tensor:")
    print_indent(scores)
    probs = softmax(scores)
    print("    attention softmax probs, shape =", probs.shape)
    print("      tensor:")
    print_indent(probs)
    out = probs @ v
    print("    attention output, shape =", out.shape)
    print("      tensor:")
    print_indent(out)
    return out


def mha(x, c_attn, c_proj, n_head, debug=False):
    if not debug:
        # fast path
        combined = linear(x, **c_attn)
        q, k, v = np.split(combined, 3, axis=-1)
        heads = []
        for i in range(n_head):
            qq = np.split(q, n_head, axis=-1)[i]
            kk = np.split(k, n_head, axis=-1)[i]
            vv = np.split(v, n_head, axis=-1)[i]
            heads.append(attention(qq, kk, vv, (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10, 0))
        merged = np.hstack(heads)
        return linear(merged, **c_proj)
    # debug path
    print("  MHA combined QKV, shape =", linear(x, **c_attn).shape)
    combined = linear(x, **c_attn)
    print("    tensor:")
    print_indent(combined)
    q, k, v = np.split(combined, 3, axis=-1)
    for name, t in zip(["Q", "K", "V"], [q, k, v]):
        print(f"  {name} split, shape =", t.shape)
        print("    tensor:")
        print_indent(t)
    heads = []
    for idx in range(n_head):
        if idx == 0:
            print(f"  head {idx}")
        qq = np.split(q, n_head, axis=-1)[idx]
        kk = np.split(k, n_head, axis=-1)[idx]
        vv = np.split(v, n_head, axis=-1)[idx]
        out = attention(qq, kk, vv, (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10, idx, debug=True)
        heads.append(out)
    merged = np.hstack(heads)
    print("  merged heads, shape =", merged.shape)
    print("    tensor:")
    print_indent(merged)
    proj = linear(merged, **c_proj)
    print("  projection, shape =", proj.shape)
    print("    tensor:")
    print_indent(proj)
    return proj


def transformer_block(x, block, n_head, debug=False):
    if not debug:
        a = mha(layer_norm(x, **block["ln_1"]), **block["attn"], n_head=n_head)
        x = x + a
        b, o = ffn(layer_norm(x, **block["ln_2"]), **block["mlp"])
        return x + o
    # debug
    print(f"\n>>> Debug transformer block")
    print("  input x, shape =", x.shape)
    print("    tensor:")
    print_indent(x)
    ln1 = layer_norm(x, **block["ln_1"])
    print("  LayerNorm1, shape =", ln1.shape)
    print("    tensor:")
    print_indent(ln1)
    attn_out = mha(ln1, **block["attn"], n_head=n_head, debug=True)
    x2 = x + attn_out
    print("  post-attention + residual, shape =", x2.shape)
    print("    tensor:")
    print_indent(x2)
    ln2 = layer_norm(x2, **block["ln_2"])
    print("  LayerNorm2, shape =", ln2.shape)
    print("    tensor:")
    print_indent(ln2)
    a, ffw_out = ffn(ln2, **block["mlp"])
    print("  FFN intermediate a, shape =", a.shape)
    print("    tensor:")
    print_indent(a)
    print("  FFN output, shape =", ffw_out.shape)
    print("    tensor:")
    print_indent(ffw_out)
    x3 = x2 + ffw_out
    print("  block output, shape =", x3.shape)
    print("    tensor:")
    print_indent(x3)
    return x3

# gpt2, generate, main below

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head, debug_layer=None):
    x = wte[inputs] + wpe[range(len(inputs))]
    for i, block in enumerate(blocks):
        is_debug = (debug_layer is not None and i == debug_layer)
        x = transformer_block(x, block, n_head, debug=is_debug)
        if is_debug:
            return
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[-n_tokens_to_generate:]


def main(
    prompt,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
    layer: int = None,
):
    from utils import load_encoder_hparams_and_params

    if isinstance(prompt, (list, tuple)):
        prompt = " ".join(str(x) for x in prompt)

    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir
    )
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + (n_tokens_to_generate or 0) < hparams["n_ctx"], \
        f"Sequence too long: {len(input_ids)} + {n_tokens_to_generate} > {hparams['n_ctx']}"

    # debug single block
    if layer is not None:
        gpt2(input_ids, params['wte'], params['wpe'], params['blocks'], params['ln_f'], hparams['n_head'], debug_layer=layer)
        return

    # normal generation
    output_ids = generate(input_ids, params, hparams['n_head'], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    print(output_text)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
