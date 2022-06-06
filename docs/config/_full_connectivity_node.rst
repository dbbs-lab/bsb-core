.. code-block:: json

  {
    "connectivity": {
      "my_conn_block": {
        "cls": "bsb.connectivity.TouchDetection",
        "presynaptic": {
          
        },
        "partitions": ["pA", "pB", "pC"],
        "overrides": {
          "my_cell_type": {
            "density": 0.002
          }
        }
      }
    }
  }
