###########
Obfuscation
###########

================================
Distributing obfuscated releases
================================

1. Obfuscate the source code

::

  cd obfuscation
  pyarmor obfuscate --recursive --output bsb ../bsb/__init__.py
  cp -r ../bsb/configurations ./bsb

2. Generate a license with expiration date

::

  pyarmor licenses --expired 2021-01-01 expires-in-2021

3. Overwrite the original license file with the generated license file.

**Generated license path:** `obfuscation/licences/expires-in-2021/license.lic`

**Original license path:** `obfuscation/bsb/pytransform/license.lic`

4. Adapt possibly outdated `obfuscation/setup_linux.py` and build the distribution.

::

  python3 setup_linux.py bdist_wheel

You now have a .whl file in `obfuscation/dist` that you can distribute as you
like.

.. note::
  For Windows, use ``setup_windows.py``
