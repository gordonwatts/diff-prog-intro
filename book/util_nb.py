from IPython.core.magic import register_cell_magic


@register_cell_magic
def write_and_run(line, cell):
    """IPython Cell Magic that will both emit cell contents to a file (even append),
    and also execute it in the notebook's context.

    `%%write_and_run [-a] filename.py`

    Write out the contents of the cell to `filename.py`, and if `-a` is present, append
    it.

    Args:
        line (str): The rest of the cell magic command - so we can grab arguments, etc.
        cell (ipython cll): Cell we are working with
    """
    argz = line.split()
    file = argz[-1]
    mode = "w"
    if len(argz) == 2 and argz[0] == "-a":
        mode = "a"
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)  # type: ignore # noqa: F821
