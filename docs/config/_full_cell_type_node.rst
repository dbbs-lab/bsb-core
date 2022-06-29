.. code-block:: json

  {
    "cell_types": {
      "my_cell_type": {
        "entity": false,
        "spatial": {
          "radius": 10.0,
          "geometry": {
            "axon_length": 150.0,
            "other_hints": "hi!"
          },
          "morphologies": [
            {
              "select": "by_name",
              "names": ["short_*"]
            },
            {
              "select": "by_name",
              "names": ["long_*"]
            }
          ]
        },
        "plotting": {
          "display_name": "Fancy Name",
          "color": "pink",
          "opacity": 1.0
        }
      }
    }
  }
