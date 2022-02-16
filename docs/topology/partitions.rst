##########
Partitions
##########

======
Voxels
======

:class:`Voxel partitions <.topology.partition.Voxels>` are an irregular shape in space,
described by a group of rhomboids, called a :class:`~.voxels.VoxelSet`. The voxel
partition needs to be configured with a :class:`~.voxels.VoxelLoader` to load the voxelset
from somewhere. Most brain atlases scan the brain in a 3D grid and publish their data in
the same way, usually in the Nearly Raw Raster Data format, NRRD. In general, whenever you
have a voxelized 3D image, a ``Voxels`` partition will help you define the shapes
contained within.

NRRD
----

To load data from NRRD files use the :class:`~.voxels.NrrdVoxelLoader`. By
default it will load all the nonzero values in a source file:

.. tabs::

   .. code-tab:: c

         int main(const int argc, const char **argv) {
           return 0;
         }

   .. code-tab:: c++

         int main(const int argc, const char **argv) {
           return 0;
         }

   .. code-tab:: py

         def main():
             return

   .. code-tab:: java

         class Main {
             public static void main(String[] args) {
             }
         }

   .. code-tab:: julia

         function main()
         end

   .. code-tab:: fortran

         PROGRAM main
         END PROGRAM main
