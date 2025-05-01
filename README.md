# GPT2 Inspector

A small CLI tool based on [picoGPT](https://github.com/jaymody/picoGPT) that can either:

- **Generate** text with GPT-2 (124M, 355M, etc.)  
- **Inspect** all intermediate tensors of a single transformer block for debugging

---

## 🚀 Quickstart


**Run with `uv`**  
- **Generate** (40-token greedy decode by default):  
 ```bash
 uv run gpt2.py "Once upon a time"
 ```
- **Specify number of tokens**:  
 ```bash
 uv run gpt2.py "Once upon a time" --n_tokens_to_generate 50
 ```
- **Debug a specific block** (print every intermediate tensor in transformer block 0):  
 ```bash
 uv run gpt2.py "Once upon a time" --layer 0
 ```

---

## ⚙️ Flags & Options

| Flag                     | Description                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------|
| `prompt` (positional)    | The text prompt to feed into GPT-2                                                   |
| `--n_tokens_to_generate` | Number of tokens to generate in normal mode (default: `40`)                          |
| `--model_size`           | Which GPT-2 size to load (`124M`, `355M`, `774M`, `1558M`)                            |
| `--models_dir`           | Path to your downloaded model files (default: `models/`)                             |
| `--layer`                | (0-based) block index to debug. If set, prints all intermediate tensors then exits    |

---

## 📄 License

MIT
