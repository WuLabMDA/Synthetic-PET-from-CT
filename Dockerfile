FROM nvidia/cuda:11.0-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 \
  curl sudo git wget htop \
  && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
ARG USER_NAME="msalehjahromi"
RUN adduser --disabled-password --gecos '' --shell /bin/bash ${USER_NAME}
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN chmod 777 /home/${USER_NAME}
WORKDIR /home/${USER_NAME}

# Install Miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && bash ~/Miniconda3-py38_4.8.2-Linux-x86_64.sh -p ~/miniconda -b \
 && rm ~/Miniconda3-py38_4.8.2-Linux-x86_64.sh
ENV PATH=/home/${USER_NAME}/miniconda/bin:$PATH
## Create a Python 3.8.3 environment
RUN /home/${USER_NAME}/miniconda/bin/conda install conda-build \
 && /home/${USER_NAME}/miniconda/bin/conda create -y --name py38 python=3.8.3 \
 && /home/${USER_NAME}/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/home/${USER_NAME}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Python packages installation
## Common packages
RUN pip install gpustat==0.6.0 setuptools==45 pytz==2021.1
RUN pip install numpy==1.16.2 scipy matplotlib==3.3.2 pandas==1.1.5
RUN pip install scikit-image==0.17.2 opencv-python==4.4.0.44 scikit-learn==0.24.1 deepdish==0.3.6 seaborn==0.11.1

## Install pytorch
RUN conda update -n base -c defaults conda
#RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN pip install tensorboard==2.4.0
RUN pip install pytorch-msssim


## Install python packages
RUN pip install dominate==2.4.0
RUN pip install visdom==0.1.8.8
RUN pip install tqdm
RUN pip install nibabel
RUN pip install SimpleITK


WORKDIR /home/${USER_NAME}


## Install nnUNet
#RUN pip install nnunet==1.6.6

# Set environment variables
#ENV HDF5_USE_FILE_LOCKING=FALSE
#ENV nnUNet_raw_data_base="/Data/nnUNet/nnUNet_raw_data_base"
#ENV nnUNet_preprocessed="/Data/nnUNet/nnUNet_preprocessed"
#ENV RESULTS_FOLDER="/Data/nnUNet/nnUNet_trained_models"



#docker build -t ctpet .

#docker run -it --rm --gpus all --shm-size=192G --user $(id -u):$(id -g) --cpuset-cpus=20-29 \
#-v /rsrch1/ip/msalehjahromi/Codes/nnUNet/pre_nnUNet:/home/msalehjahromi/pre_nnUNet \
#-v /rsrch1/ip/msalehjahromi/data:/Data \
#--name CTPET ctpet:latest