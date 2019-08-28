
# Setup
## Windows CUDA 10 Python 3.7 
python -m venv ./venv
./venv/Scripts/activate
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
pip3 install matplotlib
pip3 install scikit-image
pip3 install scipy

## Mac with anaconda python 3.7
- conda create --name py3 python=3.7
- conda activate py3
- conda install -c pytorch pytorch
- conda install -c pytorch torchvision
- conda install matplotlib
- conda install -c conda-forge scikit-image

# to run:
`python precompute_dataset.py`
- precomputes a complete dataset

Run render-style-transfer with optional boolean which dictates whether the logs are stored in a git tracked folder. 
`python render-style-transfer.py [keep_logs=False]`
- trains the model
