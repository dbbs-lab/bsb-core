git clone https://github.com/nest/nest-simulator $GITHUB_WORKSPACE/_nest_repo
cd $GITHUB_WORKSPACE/_nest_repo
git checkout tags/v$1
mkdir build
cd build
pip install cython cmake
mkdir -p $2
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$2 \
  -Dwith-mpi=ON
make install
