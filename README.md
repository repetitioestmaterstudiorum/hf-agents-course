# hf-agents-course

HuggingFace Agents Course Code.

Also revised SFT and LoRa to fine-tune LLMs to use tools.

Comparing SFT with LoRa (1000 steps) - see `sft` and `lora` folders:

![Comparing SFT with LoRa (1000 steps) - see `sft` and `lora` folders](./assets/1000-steps-sft-vs-sft-with-lora.png)

## Setup

1. Make sure you have pdm installed (macOS: `brew install pdm`).
2. `pdm install`

## Sub-Repositories

These are spaces on HuggingFace and their code is not visible on GitHub.

At the moment of documenting this, the following agents (running on HF spaces) are publicly available to try out:
- https://huggingface.co/spaces/almightyt/First_agent_template
    - Uses https://huggingface.co/docs/smolagents/reference/agents#smolagents.CodeAgent

### Sub-Repositories Setup Notes

1. Create a token with write permissions to repositories
2. Run huggingface-cli login --add-to-git-credential and enter the new token
3. `git remote set-url origin https://almightyt:<token>@huggingface.co/spaces/almightyt/<space name>`

## Useful Stuff

`watch -n 1 nvidia-smi`
