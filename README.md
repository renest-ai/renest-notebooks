# Renest Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/renest-ai/renest-notebooks/blob/main/colab/comfyui.ipynb)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

Official interactive notebooks to restore, verify, and run ComfyUI and AIGC environments on cloud notebook platforms with Renest.

---

## Available Notebooks

| Framework | Platform | Recommended Hardware | Status | Quick Launch |
| :--- | :--- | :--- | :--- | :--- |
| **ComfyUI** | Google Colab | T4 / L4 / A100 (High-RAM) | Ready | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/renest-ai/renest-notebooks/blob/main/colab/comfyui.ipynb) |

---

## How It Works

1. **Restore with Hand-off Code**:
   Paste a Hand-off code from an official example or your community/team. The notebook runs `renest restore`, which performs hardware inspection, downloads large model weights, reconstructs the isolated virtual environment via `uv`, and validates byte integrity.
2. **Launch Supervised WebUI**:
   Start ComfyUI and open an encrypted public Cloudflare tunnel (`*.trycloudflare.com`). The supervisor manages both the ComfyUI process and the tunnel, streaming real-time generation progress and ensuring zero orphaned processes on shutdown.
3. **Modify & Save Back**:
   Tweak prompts and test custom nodes directly in the WebUI. Save modified workflows back to your Renest drive through the device login flow or via `renest pack`.

---

## Directory Structure

```text
renest-notebooks/
├── colab/
│   ├── comfyui.ipynb       # ComfyUI one-click restore & WebUI launcher
│   ├── renest_colab.py     # Supervisor managing ComfyUI & tunnel lifecycle
│   └── README.md           # Colab-specific usage guide
├── LICENSE                 # Apache-2.0
└── README.md               # Repository showcase & entry points
```

---

## Security & Isolation

- **Zero Credential Storage**: Notebooks do not store master cloud keys or long-lived authentication tokens. Runtimes only hold scoped, ephemeral grant tokens.
- **Fail-Fast Integrity**: Package restore verifies sha256 payloads before execution. If dependencies cannot be cleanly reconstructed, it fails fast rather than serving a broken runtime.
- **Zero Process Leaks**: The runner traps termination signals (SIGINT/SIGTERM) and systematically reaps child process groups to avoid lingering background compute consumption.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
