.. code-block:: json

  {
    "connectivity": {
      "my_conn_block": {
        "strategy": "bsb.connectivity.VoxelIntersection",
        "presynaptic": {
          "cell_types": ["A", "B"]
        },
        "postsynaptic": {
          "cell_types": ["C", "D"]
        }
      }
    }
  }
