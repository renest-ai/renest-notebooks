# Renest × Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/renest-ai/renest-notebooks/blob/main/colab/comfyui.ipynb)

This module provides Google Colab users with a clean workflow to restore verified Renest asset packages, perform environment verification, and access the ComfyUI WebUI.

## Principles

1. **Standard Client Flow**:
   - Executes official `renest restore` with multi-stage verification (hardware check, asset transfer, byte verification, uv dependency reconstruction, and launch verification).
2. **WebUI Access via Tunnel**:
   - Launches ComfyUI and uses a Cloudflare tunnel (`cloudflared`) to provide a public temporary access link (`https://*.trycloudflare.com`).
   - Enables browser-based interaction on Google Colab without manual port forwarding.
3. **Clean Lifecycle & Zero Leaks**:
   - Host supervisor ensures termination of both child processes (ComfyUI and cloudflared) on interruption or exit.
   - Preserves workspace isolation and prevents process leaks.

## Files

* [`comfyui.ipynb`](./comfyui.ipynb): Standalone two-cell Jupyter Notebook for Google Colab.
* [`renest_colab.py`](./renest_colab.py): Standalone supervisor managing `renest start` and Cloudflare tunnel lifecycle.

## Quickstart

1. Open [`comfyui.ipynb`](https://colab.research.google.com/github/renest-ai/renest-notebooks/blob/main/colab/comfyui.ipynb) in Google Colab.
2. Select a GPU runtime (T4, L4, or A100). High-RAM is recommended.
3. Paste your **Hand-off code** into Step 1 and run the cell.
4. Run Step 2 to launch ComfyUI and open the generated public WebUI URL in your browser.
