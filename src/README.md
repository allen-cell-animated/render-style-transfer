
# Windows CUDA 10 Python 3.7 
python -m venv ./venv
./venv/Scripts/activate
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
pip3 install matplotlib
pip3 install scikit-image
pip3 install scipy

# to run:
python precompute_dataset.py
- precomputes a complete dataset

python render-style-transfer.py
- trains the model
