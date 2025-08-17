export CUDA_HOME=/usr/lib/cuda
export CUDA_PATH=/usr/lib/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
sudo mkdir -p /usr/local/cuda/nvvm/libdevice
sudo ln -s /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
nvcc --version
pip uninstall dlib -y
pip install dlib --verbose --force-reinstall
