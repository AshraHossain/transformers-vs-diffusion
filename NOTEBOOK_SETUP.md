# Running Notebooks: Setup Guide

All notebooks have automatic path setup in the first cell, so they work from any directory. The notebooks demonstrate the thesis: identical shared modules (MultiHeadSelfAttention, TokenPositionalEmbedding) configured differently create two distinct generation paradigms.

## Quick Start

### Option 1: Run from Project Root (Recommended)

```bash
cd /path/to/transformers-vs-diffusion
jupyter notebook notebooks/01_attention.ipynb
```

### Option 2: Run from Notebooks Directory

```bash
cd /path/to/transformers-vs-diffusion/notebooks
jupyter notebook 01_attention.ipynb
```

Both work because each notebook has automatic path setup in the first cell.

## Why the Setup Cell?

The first cell of each notebook contains:

```python
import sys
from pathlib import Path

# Handles multiple working directory scenarios
current_file = Path(__file__).resolve() if '__file__' in dir() else Path.cwd()
if current_file.is_file():
    notebook_dir = current_file.parent
    project_root = notebook_dir.parent
else:
    project_root = Path.cwd()
    if project_root.name == 'notebooks':
        project_root = project_root.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

This ensures that `src` module is importable regardless of where you run the notebook from.

## Troubleshooting

### Still getting "ModuleNotFoundError: No module named 'src'"?

1. **Verify project structure:**
   ```bash
   ls -la src/
   # Should show: common/ transformer/ diffusion/
   ```

2. **Check Jupyter working directory:**
   - In the first cell of any notebook, run:
   ```python
   import os
   print(os.getcwd())
   # Should show project root or notebooks/ directory
   ```

3. **Manually set path (if needed):**
   - Add this before any `from src` imports:
   ```python
   import sys
   sys.path.insert(0, '/full/path/to/transformers-vs-diffusion')
   ```

4. **Restart kernel:**
   - Jupyter → Kernel → Restart
   - This clears cached imports

### Import still fails?

The path setup cell runs automatically, but if you need to debug:

```python
# In notebook cell:
import sys
print("sys.path:", sys.path[:3])
from pathlib import Path
print("src exists:", (Path.cwd().parent / 'src').exists())
```

## Notebooks Overview

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| 01_attention.ipynb | Understand attention mechanism | ~2 min |
| 02_gpt.ipynb | Train and generate with GPT | ~1 min |
| 03_diffusion_lm.ipynb | Train and generate masked LM | ~1 min |
| 04_comparison.ipynb | Side-by-side comparison | ~3 min |

## Environment

- Python 3.11+
- PyTorch 2.0+
- Jupyter Notebook

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running All Notebooks

```bash
# From project root:
jupyter notebook notebooks/
```

Then open each notebook in the Jupyter interface.

## Notes

- First cell of each notebook handles imports automatically
- No additional setup needed beyond having Jupyter installed
- All notebooks are self-contained (no external data dependencies)
- Toy corpus is created in each notebook, no data downloads needed
