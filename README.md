# hf_AC

## Environment Setup
- Python 3.9+
- PyTorch **2.5.1+** and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/, pip install recommended)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
git clone https://github.com/ff2416/hf_AC.git
cd hf_AC
pip install -e .
```
## Model Installation
https://huggingface.co/FF2416/AC-Foley/blob/main/model.pth

## Inference
```bash
python inf.sh \
  --model_path <model path> \
  --duration 8 \
  --prompt <prompt> \
  --video_dir <videos directory or video path> \
  --audio_path <audio path> \
  --output <output path>
```
