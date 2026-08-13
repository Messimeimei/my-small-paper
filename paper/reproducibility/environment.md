# Paper Environment

## Writing and Analysis

- OS: Ubuntu 24.04 (WSL environment)
- Python environment: `../.venv-paper`
- Python dependency lock: `analysis/requirements-lock.txt`
- LaTeX compiler: project-local TeX Live 2026 under `../.tools/texlive/2026/`
- Default engine: pdfLaTeX through `latexmk`
- Fallback engine retained locally: Tectonic 0.17.0
- PDF preflight tools: project-local Poppler 24.02.0 under `../.tools/poppler/`
- ACL template provenance: `manuscript/TEMPLATE_PROVENANCE.md`

## Training and Evaluation

Record the exact CUDA, GPU, Transformers, TRL, PEFT, PyTorch, vLLM, model, and
checkpoint versions used by final experiments. Do not infer these from the
paper-writing environment.
