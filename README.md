# Bi-Mamba-Freq Experiments Code
```bash
conda create -n mamba python=3.9 -y
conda activate mamba
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy matplotlib scikit-learn scipy jupyter wandb dvc pyarrow hydra-core openai transformers huggingface_hub
pip install sumy --no-build-isolation
cd mamba.py
pip install -e .
cd ..
```
