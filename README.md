# hf-agents-course

HuggingFace Agents Course Code.

Also revised SFT and LoRa to fine-tune LLMs to use tools.

Comparing SFT with LoRa (1000 steps) - see `sft` and `lora` folders:

![Comparing SFT with LoRa (1000 steps) - see `sft` and `lora` folders](./assets/1000-steps-sft-vs-sft-with-lora.png)

## Setup

- Make sure you have pdm installed (macOS: `brew install pdm`).
- `pdm python install 3.12`
- `pdm install` # this creates a .venv that can be selected to execute notebooks
- Execute notebooks or run Python files using pdm `pdm run <filename>`

## Sub-Repositories

These are spaces on HuggingFace and their code is not visible on GitHub.

### Sub-Repositories Setup Notes

Check sub-repo READMEs.

## Useful Stuff

`watch -n 1 nvidia-smi`
