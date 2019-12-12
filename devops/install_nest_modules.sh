PREVIOUS_DIR=$PWD
cd /home/travis
export NEST_INSTALL_DIR=/home/travis/nest-$NEST_VERSION
git clone https://github.com/dbbs-lab/cereb-nest/
cd cereb-nest
mkdir build
cd build
CEREBNEST_BUILD_DIR=$PWD
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ..
export NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest:$NEST_MODULE_PATH
export SLI_PATH=${NEST_INSTALL_DIR}/share/nest/sli:$SLI_PATH
make
make install
cd $PREVIOUS_DIR
