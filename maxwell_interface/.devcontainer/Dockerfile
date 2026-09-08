FROM dealii/dealii:v9.5.0-jammy
USER root
RUN  echo 'dealii:dealii' | chpasswd
EXPOSE 22
RUN service ssh start
RUN ["apt-get", "-y", "update"]

# LLVM
RUN ["apt-get", "-y", "install", "clang"]
RUN ["apt-get", "-y", "install", "lldb"]
RUN ["apt-get", "-y", "install", "lld"]

# UTILS
RUN ["apt-get", "-y", "install", "doxygen"]
RUN ["apt-get", "-y", "install", "gdb"]
RUN ["apt-get", "-y", "install", "git"]

# SCIENCE TOOLING
RUN ["apt-get", "-y", "install", "fftw-dev"]
RUN ["apt-get", "-y", "install", "libfftw3-3"]
RUN ["apt-get", "-y", "install", "gmsh"]

# GMSH OpenMP support
# WORKDIR /opt/gmsh_install_dir
# RUN ["git", "clone", "https://gitlab.onelab.info/gmsh/gmsh.git"]
# WORKDIR /opt/gmsh_install_dir/gmsh
# RUN ["git", "pull"]
# RUN ["mkdir", "build"]
# WORKDIR /opt/gmsh_install_dir/gmsh/build
# RUN ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_OPENMP=1"]
# RUN ["make", "install", "-j20"]


USER dealii
WORKDIR /home/dealii