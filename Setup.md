# Setup enviroment

## Which version of Python is supported?
Python 3.6.4

## Which version of Chainer is supported?
Chainer 4.1.0

## Install pyenv
```
$ git clone https://github.com/yyuu/pyenv.git ~/.pyenv
$ git clone git://github.com/yyuu/pyenv-update.git ~/.pyenv/plugins/pyenv-update
$ git clone https://github.com/yyuu/pyenv-pip-rehash.git ~/.pyenv/plugins/pyenv-pip-rehash
$ git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
```

## Install Anaconda3
```
$ pyenv install anaconda3-5.1.0
$ cd ./adversarial_text/
$ pyenv local anaconda3-5.1.0
```

## Install numpy/cython/cudnnenv
```
$ pip install --upgrade pip
$ pip install numpy scipy six Cython h5py cudnnenv
$ cudnnenv install v6-cuda8
$ cudnnenv activate v6-cuda8
```

## Set env
```
$ vim ~/.bashrc

# cuda (please change your enviroment)
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-8.0
export CPATH=/usr/local/cuda-8.0/include:$CPATH
export CUDA_PATH=/usr/local/cuda-8.0
# cudnnenv (pip install cudnnenv)
export LD_LIBRARY_PATH=~/.cudnn/active/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=~/.cudnn/active/cuda/include:$CPATH
export LIBRARY_PATH=~/.cudnn/active/cuda/lib64
```

## Install Chainer/Cupy
```
$ pip install chainer --no-cache-dir
$ pip install cupy --no-cache-dir
# it will takes 10 minutes...
```
