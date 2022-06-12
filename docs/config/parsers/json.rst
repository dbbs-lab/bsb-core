###########
JSON parser
###########

The JSON parser is built on top of Python's `json
<https://docs.python.org/3/library/json.html>`_ module  and adds 2 additional features:

* JSON references
* JSON imports

.. _json_ref:

===============
JSON References
===============

References point to another JSON dictionary somewhere in the same or another document and
copy over that dictionary into the parent of the reference statement:

.. code-block:: json

  {
    "template": {
      "A": "value",
      "B": "value"
    },
    "copy": {
      "$ref": "#/template"
    }
  }

Will be parsed into:

.. code-block:: json

  {
    "template": {
      "A": "value",
      "B": "value"
    },
    "copy": {
      "A": "value",
      "B": "value"
    }
  }

.. note::

	Imported keys will not override keys that are already present. This way you can specify
	local data to customize what you import. If both keys are dictionaries, they are merged;
	with again priority for the local data.

Reference statement
===================

The reference statement consists of the :guilabel:`$ref` key and a 2-part value. The first
part of the statement before the ``#`` is the ``document``-clause and the second part the
``reference``-clause. If the ``#`` is omitted the entire value is considered a
``reference``-clause.

The document clause can be empty or omitted for same document references. When a document
clause is given it can be an absolute or relative path to another JSON document.

The reference clause must be a JSON path, either absolute or relative to a JSON
dictionary. JSON paths use the ``/`` to traverse a JSON document:

.. code-block:: json

  {
    "walk": {
      "down": {
        "the": {
          "path": {}
        }
      }
    }
  }

In this document the deepest JSON path is ``/walk/down/the/path``.

.. warning::

    Pay attention to the initial ``/`` of the reference clause! Without it, you're making
    a reference relative to the current position. With an initial ``/`` you make a
    reference absolute to the root of the document.

.. _json_import:

============
JSON Imports
============

Imports are the bigger cousin of the reference. They can import multiple dictionaries from
a common parent at the same time as siblings:

.. code-block:: json

  {
    "target": {
      "A": "value",
      "B": "value",
      "C": "value"
    },
    "parent": {
      "D": "value",
      "$import": {
        "ref": "#/target",
        "values": ["A", "C"]
      }
    }
  }

Will be parsed into:

.. code-block:: json

  {
    "target": {
      "A": "value",
      "B": "value",
      "C": "value"
    },
    "parent": {
      "A": "value",
      "C": "value"
    }
  }

.. note::

	If you don't specify any :guilabel:`values` all nodes will be imported.

.. note::

	The same merging rules apply as to the reference.

The import statement
====================

The import statement consists of the :guilabel:`$import` key and a dictionary with 2 keys:

* The :guilabel:`ref` key (note there's no ``$``) which will be treated as a reference
  statement. And used to point at the import's reference target.
* The :guilabel:`values` key which lists which keys to import from the reference target.
