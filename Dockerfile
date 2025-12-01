# Define image with OpenMDAO/Dymos/Aviary and OpenVSP support

FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Install updates
RUN apt-get update -y && apt-get install \
    python3-dev python3-sphinx vim nano bash-completion unzip wget file desktop-file-utils git sudo make cmake=3.28.3-1build7 swig g++ gfortran doxygen graphviz texlive-latex-base \
    libblas-dev liblapack-dev libxml2-dev libfltk1.3-dev libcpptest-dev libjpeg-dev libglm-dev libeigen3-dev libcminpack-dev libglew-dev -y

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb ;\
    apt-get install -y ./google-chrome-stable_current_amd64.deb

# Create user
ENV USER=omdao
RUN adduser --shell /bin/bash --disabled-password ${USER}
RUN adduser ${USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# Install Miniforge
RUN wget -q -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" ;\
    bash Miniforge3.sh -b ;\
    rm Miniforge3.sh ;\
    export PATH=$HOME/miniforge3/bin:$PATH ;\
    conda init bash

# Create conda environment
RUN source $HOME/miniforge3/etc/profile.d/conda.sh ;\
    conda create -n mdaowork python=3.11 numpy scipy cython swig -y

# Modify .bashrc
RUN echo "## Always activate mdaowork environment on startup ##" >> ~/.bashrc ;\
    echo "conda activate mdaowork" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc ;\
    echo "## OpenMPI settings" >> ~/.bashrc ;\
    echo "export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe" >> ~/.bashrc ;\
    echo "export OMPI_MCA_rmaps_base_oversubscribe=1" >> ~/.bashrc ;\
    echo "export OMPI_MCA_btl=^openib" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc ;\
    echo "## Required for some newer MPI / libfabric instances" >> ~/.bashrc ;\
    echo "export RDMAV_FORK_SAFE=true" >> ~/.bashrc ;\
    echo "" >> ~/.bashrc

# Build and Install OpenVSP (based on script from Irian Ordaz @ LARC)
RUN source $HOME/miniforge3/etc/profile.d/conda.sh ;\
    conda activate mdaowork ;\
    mkdir OpenVSP && cd OpenVSP ;\
    git clone https://github.com/OpenVSP/OpenVSP.git repo ;\
    cd repo ;\
    git checkout tags/OpenVSP_3.41.1 -b OpenVSP_3.41.1 ;\
    cd .. ;\
    mkdir -p build/Libraries build/vsp ;\
    cd build/Libraries ;\
    cmake -DCMAKE_BUILD_TYPE=Release -DVSP_NO_GRAPHICS=true ../../repo/Libraries ;\
    make ;\
    cd ../vsp ;\
    cmake -DCMAKE_BUILD_TYPE=Release -DVSP_NO_GRAPHICS=true \
          -DPYTHON_LIBRARY=$HOME/miniforge3/envs/mdaowork/lib/libpython3.so \
          -DPYTHON_INCLUDE_DIR=$HOME/miniforge3/envs/mdaowork/include/python3.11 \
          -DVSP_LIBRARY_PATH=$HOME/OpenVSP/build/Libraries \
          ../../repo/src ;\
    make ;\
    cd vsp ;\
    make;\
    sudo make install

# Install OpenMDAO/Dymos/Aviary into conda environment
RUN source $HOME/miniforge3/etc/profile.d/conda.sh ;\
    conda activate mdaowork ;\
    #
    # Install OpenMDAO/Dymos/Aviary dependencies and OpenVSP Python API
    #
    conda install matplotlib graphviz -q -y ;\
    conda install mpi4py openmpi petsc4py=3.20 pyoptsparse -q -y ;\
    python -m pip install pyparsing psutil objgraph plotly pyxdsm pydot ;\
    #
    # Install build_pyoptsparse
    # (this will allow the user additional options for installing pyoptsparse, beyond the conda install above)
    #
    python -m pip install git+https://github.com/openmdao/build_pyoptsparse ;\
    # build_pyoptsparse -v ;\
    #
    # Install OpenMDAO
    #
    git clone https://github.com/OpenMDAO/OpenMDAO.git ;\
    python -m pip install -e 'OpenMDAO[all]' ;\
    #
    # Install Dymos
    #
    git clone https://github.com/OpenMDAO/dymos.git ;\
    python -m pip install -e 'dymos[all]' ;\
    #
    # Install Aviary
    #
    git clone https://github.com/OpenMDAO/Aviary.git ;\
    python -m pip install -e 'Aviary[all]' ;\
    #
    # Install MPhys
    #
    git clone https://github.com/OpenMDAO/MPhys.git ;\
    python -m pip install -e 'MPhys[all]' ;\
    #
    # Install OpenAeroStruct
    #
    git clone https://github.com/mdolab/OpenAeroStruct.git ;\
    python -m pip install -e 'OpenAeroStruct' ;\
    #
    # Install OpenVSP Python support
    #
    conda install wxPython ;\
    cd $HOME/OpenVSP/build/vsp/python_pseudo ;\
    pip install -r requirements-dev.txt

# Set up a work directory that can be shared with the host operating system
WORKDIR /home/${USER}/work

ENTRYPOINT ["tail", "-f", "/dev/null"]
