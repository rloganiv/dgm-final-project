#! /bin/bash
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi


# Deactivate the travis-provided venv and setup a conda-based venv
deactivate

# Add mini conda to the path
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use miniconda to setup conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -q -n testenv --yes python="$PYTHON_VERSION"
fi
cd ..
popd

# Activate the environment
echo "Activating Environment"
source activate testenv

# Install requirements via pip in our conda environment
echo "Installing requirements"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
