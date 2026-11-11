"""Names shown to the user.

Files and folders keep the short ids (group03, group03_plant01). Everything the
user reads shows them as "plant_cluster_03" and "plant_instance_01"
(one id: "plant_cluster_03_plant_instance_01").
"""

from __future__ import annotations

import re

_PAIR = re.compile(r"group(\d+)_plant(\d+)")
_GROUP = re.compile(r"\bgroup(\d+)")
_PLANT = re.compile(r"\bplant(\d+)")


def cluster(g: str) -> str:
    """'group03' -> 'plant_cluster_03'."""
    return _GROUP.sub(r"plant_cluster_\1", g)


def instance(pid: str) -> str:
    """'group03_plant01' or 'plant01' -> 'plant_instance_01'."""
    m = _PAIR.search(pid) or _PLANT.search(pid)
    return f"plant_instance_{m.groups()[-1]}" if m else pid


def nice(text: str) -> str:
    """Replace every id in a text: 'group03_plant01' -> 'plant_cluster_03_plant_instance_01'."""
    text = _PAIR.sub(r"plant_cluster_\1_plant_instance_\2", str(text))
    return _PLANT.sub(r"plant_instance_\1", _GROUP.sub(r"plant_cluster_\1", text))


def raw(name: str) -> str:
    """Inverse of nice() for one id: 'plant_cluster_03_plant_instance_01' -> 'group03_plant01'."""
    name = re.sub(r"plant_cluster_(\d+)_plant_instance_(\d+)", r"group\1_plant\2", name)
    return re.sub(r"plant_instance_(\d+)", r"plant\1", re.sub(r"plant_cluster_(\d+)", r"group\1", name))


# column names of the csv tables, as shown in the report and the app
COLUMN = {"plant_id": "plant instance", "group": "plant cluster", "n_groups": "plant clusters",
          "matched_plant": "matched plant instance", "n_plants": "plant instances"}


def column(c: str) -> str:
    return COLUMN.get(c, c)
