#!/usr/bin/env sh

export NEST_INSTALL_DIR=/home/travis/nest-$NEST_VERSION
python --version
python -m install --upgrade pip
python -m pip install neuron
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
python -c "import nest; nest.Install('cerebmodule')"
python -c "import neuron; neuron.test()"
python -m install -r requirements.txt
python -m install coverage
python -m install -e .
