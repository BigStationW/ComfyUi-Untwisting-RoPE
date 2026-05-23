# ComfyUi-Untwisting-RoPE (Training-Free Style Transfer)

This is a ComfyUi implementation of [Untwisting RoPE: Frequency Control for Shared Attention in DiTs](https://arxiv.org/abs/2602.05013)

https://untwisting-rope.github.io/

<img width="1280" alt="Image 4" src="https://github.com/user-attachments/assets/ea4b7f49-45b0-4812-845b-0bd58824f27c" />


- Only Z-image Turbo is implemented so far (I'll see if I'll implement other models).
- You can see more examples [here](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE/tree/main/Examples).

## Installation

Navigate to the **ComfyUI\custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUi-Untwisting-RoPE
```
Restart ComfyUI after installation.

## Usage

Here's a [workflow](https://github.com/BigStationW/ComfyUi-Untwisting-RoPE/blob/main/Workflow_zimage_turbo.json) for those interested.
You also need this [custom node](https://github.com/BigStationW/ComfyUi-Scale-Image-to-Total-Pixels-Advanced) to make it work.
