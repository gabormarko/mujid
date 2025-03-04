FROM ubuntu:jammy

# Create a non-root user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN if getent passwd ${USER_UID}; then \
    userdel -r $(getent passwd ${USER_UID} | cut -d: -f1); \
    fi

# Delete existing group if it exists
RUN if getent group ${USER_GID}; then \
    groupdel $(getent group ${USER_GID} | cut -d: -f1); \
    fi

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    zsh \
    curl \
    vim \
    build-essential \
    cmake \
    wget \
    git \
    unzip \
    pip \
    cmake \
    libpoco-dev \
    libeigen3-dev \
    gnupg \
    mesa-utils \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libgl1-mesa-gl \
    dpkg

# Add robotpkg repository
RUN apt-get update && apt-get install -y lsb-release \
    && echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - \
    && apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    robotpkg-py310-tsid

# Set environment variables
ENV PATH=/opt/openrobots/bin:$PATH \
    PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH \
    LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH \
    CMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake:$CMAKE_PREFIX_PATH



# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

USER $USERNAME

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

SHELL ["/bin/zsh", "-c"]
