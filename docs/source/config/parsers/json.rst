###############
BSB JSON parser
###############

The BSB's JSON parser sports 2 additional features over the builtin Python JSON parser:

* JSON references
* JSON imports

===============
JSON References
===============

References point to another JSON dictionary somewhere in the same or another document and
copy over that dictionary into the parent of the reference statement:

.. code-block:: json

  {
    "target": {
      "A": "value",
      "B": "value"
    },
    "parent": {
      "$ref": "#/target"
    }
  }

Will be parsed into:

.. code-block:: json

  {
    "target": {
      "A": "value",
      "B": "value"
    },
    "parent": {
      "A": "value",
      "B": "value"
    }
  }

.. note::

	The data that you import/reference will be combined with the data that's already present
	in the parent. The data that is already present in the parent will overwrite keys that
	are imported. In the special case that the import and original both specify a dictionary
	both dictionaries' keys will be merged, with again (and recursively) the original data
	overwriting the imported data.

Reference statement
===================

The reference statement consists of the ``$ref`` key and a 2-part value. The first part of
the statement before the ``#`` is the ``document``-clause and the second part the
``reference``-clause. If the ``#`` is omitted the entire value is considered a
``reference``-clause.

The document clause can be empty or omitted and the reference will point to somewhere
within the same document. When a document clause is given it can be an absolute or
relative path to another JSON document.

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

Where the deepest node could be accessed with the JSON path ``/walk/down/the/path``.

.. warning::

	Relative reference clauses are valid! It's easy to forget the initial ``/`` of a
	reference clause! Take ``other_doc.json#some/path`` as example. If this reference is
	given from ``my/own/path`` then you'll be looking for ``my/own/path/some/path`` in the
	other document!

============
JSON Imports
============

Imports are the bigger cousin of the reference. They can import multiple dictionaries from
a common parent at the same time. Where the reference would only be able to import either
the whole parent or a single child, the import can selectively pick children to copy as
siblings:

.. code-block:: json

  {
    "target": {
      "A": "value",
      "B": "value",
      "C": "value"
    },
    "parent": {
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

	The data that you import/reference will be combined with the data that's already present
	in the parent. The data that is already present in the parent will overwrite keys that
	are imported. In the special case that the import and original both specify a dictionary
	both dictionaries' keys will be merged, with again (and recursively) the original data
	overwriting the imported data.

The import statement
====================

The import statement consists of the ``$import`` key and a dictionary with 2 keys:

* The ``ref`` key (note there's no ``$``) which will be treated as a reference statement.
  And used to point at the import's reference target.
* The ``value`` key which lists which keys to import from the reference target.
