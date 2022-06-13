.. code-block:: json

  {
    "placement": {
      "my_placement_block": {
        "strategy": "bsb.placement.RandomPlacement",
        "cell_types": ["A", "B", "C"],
        "partitions": ["pA", "pB", "pC"],
        "overrides": {
          "my_cell_type": {
            "density": 0.002
          }
        }
      }
    }
  }
