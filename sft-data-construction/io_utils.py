import json
import io
import os
# import datasets
# ds=datasets.arrow_dataset.Dataset()
# ds.to_json()


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    with _make_w_io_base(f, mode) as fw:
        if isinstance(obj, list):
            fw.writelines(map(lambda x: json.dumps(x,ensure_ascii=False)+"\n", obj ) )
        elif isinstance(obj, dict):
            json.dump(obj, f, indent=indent, default=default)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    with _make_r_io_base(f, mode) as f:
        records = []
        for line in f:
            records.append(json.loads(line))
    return records
