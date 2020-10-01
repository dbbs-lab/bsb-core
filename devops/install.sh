#!/usr/bin/env sh

export NEST_INSTALL_DIR=/home/travis/nest-$NEST_VERSION
wget https://neuron.yale.edu/ftp/neuron/versions/v7.7/nrn-7.7.x86_64-linux.deb
sudo dpkg -i nrn-7.7.x86_64-linux.deb
export PYTHONPATH=/usr/local/nrn/lib/python:$PYTHONPATH
source devops/check_nest_cache.sh
sudo apt-get install -y
if [ "$HAS_NEST_CACHE" = "0" ]; then
  source devops/install_nest.sh
else
  echo "NEST cache found, skipping installation";
fi
if [ "$HAS_CEREBNEST_CACHE" = "0" ]; then
  source devops/install_nest_modules.sh
else
  echo "CEREBNEST cache found, skipping installation";
fi
source devops/post_install_env_vars.sh
python -c "import nest; nest.Install('cerebbmodule')"
pip install --upgrade pip
pip install -r requirements.txt
pip install coverage
pip install -e .
