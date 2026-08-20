FROM osrf/ros:noetic-desktop-full

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ─── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    coinor-libipopt-dev \
    gfortran \
    liblapack-dev \
    libatlas-base-dev \
    pkg-config \
    swig \
    python3-pip \
    python3-scipy \
    python3-tk \
    python3-catkin-tools \
    git \
    vim \
    wget \
    curl \
    # ROS packages
    ros-noetic-turtlebot3-msgs \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    # X11 / display
    x11-apps \
    x11-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxrender1 \
    libxext6 \
    libxau6 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ─── CasADi 3.7.0 (compiled from source with IPOPT + Python bindings) ───────
RUN git clone --depth 1 --branch 3.7.0 https://github.com/casadi/casadi.git /tmp/casadi \
    && cd /tmp/casadi \
    && mkdir build && cd build \
    && cmake \
        -DWITH_PYTHON=ON \
        -DWITH_PYTHON3=ON \
        -DWITH_IPOPT=ON \
        -DWITH_OPENMP=ON \
        -DWITH_THREAD=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/casadi

# ─── Python dependencies ─────────────────────────────────────────────────────
# POT 0.9.3 is required by the ot_process planning module
RUN pip3 install --no-cache-dir "pot==0.9.3"

# ─── Install f1tenth_gym (build context = /home/mrvik, so f1tenth_gym is accessible)
# Maps are mounted as a volume at runtime so custom maps stay live-editable.
COPY f1tenth_gym /opt/f1tenth_gym
# Install f1tenth_gym runtime deps only (NOT the package itself).
# f1tenth_gym is imported directly from /opt/f1tenth_gym via PYTHONPATH so
# maps are resolved correctly relative to the source tree.
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir \
        "numpy>=1.18,<2.0" \
        "gymnasium>=0.29.1,<1.0" \
        "pygame>=2.5.2" \
        "numba>=0.55.2" \
        "shapely>=2.0.2" \
        "Pillow>=9.0.1" \
        "opencv-python-headless>=4.9.0" \
        "PyYAML>=5.3.1" \
        "scipy>=1.7.3" \
        "requests>=2.31.0"

# Make f1tenth_gym importable from its source tree (maps stay in /opt/f1tenth_gym/maps/)
ENV PYTHONPATH=/opt/f1tenth_gym:${PYTHONPATH}

# ─── Clone DEFORM repository ─────────────────────────────────────────────────
RUN git clone https://github.com/NeSC-IV/DEFORM.git /root/DEFORM

# ─── F1Tenth: raise NMPC limits (stock DEFORM caps at 0.22 m/s for TurtleBot3) ─
RUN sed -i \
    's/#define MAX_LINEAR_VELOCITY     0.22/#define MAX_LINEAR_VELOCITY     3.0/' \
    /root/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h \
    && sed -i \
    's/#define MAX_ANGULAR_VELOCITY    2.0/#define MAX_ANGULAR_VELOCITY    2.5/' \
    /root/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h \
    && sed -i \
    's/#define MAX_DELTA_VELOCITY    1.7/#define MAX_DELTA_VELOCITY      3.0/' \
    /root/DEFORM/src/control/traj_opti/include/traj_opti/traj_opti.h

# ─── Copy f1tenth_deform_bridge ROS package into the workspace ───────────────
COPY deform_docker/f1tenth_deform_bridge /root/DEFORM/src/utility/f1tenth_deform_bridge

# ─── Build the catkin workspace (DEFORM + bridge package) ────────────────────
# DEFORM repo root is itself a catkin workspace (contains src/)
RUN /bin/bash -c "\
    source /opt/ros/noetic/setup.bash && \
    cd /root/DEFORM && \
    catkin_make \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DCMAKE_BUILD_TYPE=Release \
    "

# ─── Shell environment ────────────────────────────────────────────────────────
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /root/DEFORM/devel/setup.bash" >> /root/.bashrc \
    && echo "export TURTLEBOT3_MODEL=burger" >> /root/.bashrc \
    && echo "export QT_X11_NO_MITSHM=1" >> /root/.bashrc \
    && echo "export LIBGL_ALWAYS_SOFTWARE=0" >> /root/.bashrc

# ─── X11 / display environment variables ─────────────────────────────────────
# DISPLAY is intentionally left unset here; mount it from the host via run.sh
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV SDL_VIDEODRIVER=x11

# ─── Entrypoint ───────────────────────────────────────────────────────────────
# ros_entrypoint.sh sources /opt/ros/noetic/setup.bash automatically
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
