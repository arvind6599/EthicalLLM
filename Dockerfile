# inspired from: https://gist.github.com/mpagli/d9550430b6beead3c1747f6cfa7b4ab7

# Use the Nvidia PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.10-py3


# Set environment variable to avoid interactive time zone configuration
ENV DEBIAN_FRONTEND=noninteractive

# Set the time zone to your desired value, e.g., "Etc/UTC"
ENV TZ=Etc/UTC

# Install git using the Alpine image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install some necessary tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common
# RUN apt-get install -y \
#         inkscape \
#         texlive-latex-extra \
#         dvipng \
#         texlive-full \
#         jed \
#         libsm6 \
#         libxext-dev \
#         libxrender1 \
#         lmodern \
#         libcurl3-dev \
#         libfreetype6-dev \
#         libzmq3-dev \
#         libcupti-dev \
#         pkg-config \
#         libjpeg-dev \
#         libpng-dev \
#         zlib1g-dev \
#         locales
RUN apt-get install -y \
        zsh \
        git \
        screen \
        tmux  \ 
        python3-pip \
        python3-venv

# RUN apt-get install -y openssh-server
# # install good vim.
# RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
# RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=amenon \
    NB_UID=255584 \
    NB_GROUP=runai-dhlab \
    NB_GID=30207 
ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}


# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

###### switch to user and compile test example.
USER ${NB_USER}

###### switch to root
# expose port for ssh and start ssh service.
EXPOSE 22
# expose port for notebook.
EXPOSE 8888
# expose port for tensorboard.
EXPOSE 6666