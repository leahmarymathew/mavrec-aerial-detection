"""Run ONE MAVREC notebook for real (SMOKE_TEST stays False) inside the current Colab runtime.

Used from a cell / scratch cell of a runtime that already has Drive mounted (e.g. Run_Smoke_All):

    exec(open("/content/drive/MyDrive/MAVREC/Final_Project_Code/run_full.py").read())
    run_full("01_Screening_YOLO")

The notebook executes unchanged in a fresh kernel on the same VM, except for one first cell that reuses this
runtime's Drive mount (no second consent prompt / no second Colab session). The executed copy is saved after EVERY
cell to Final_Project_Results/executed_full/<name>.ipynb, so progress is visible and survives a disconnect.
If Colab disconnects: reconnect, re-run the two lines above — finished experiments are skipped and training resumes
from the last checkpoint on Drive.
"""
import os, re, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_DIR = "/content/drive/MyDrive/MAVREC/Notebooks"
OUT_DIR = "/content/drive/MyDrive/MAVREC/Final_Project_Results/executed_full"
PRELUDE = '''# [added by run_full] Drive is already mounted in this VM: reuse it, do not ask again
import os
from google.colab import drive as _drive
if os.path.isdir("/content/drive/MyDrive"):
    _drive.mount = lambda *a, **k: print("Drive already mounted (run_full)")'''


def run_full(name, nb_dir=NB_DIR, out_dir=OUT_DIR, prelude=PRELUDE, cwd="/content"):
    os.makedirs(out_dir, exist_ok=True)
    nb = nbformat.read(os.path.join(nb_dir, name + ".ipynb"), as_version=4)
    setup = next(c for c in nb.cells if c.cell_type == "code")
    assert re.search(r"^SMOKE_TEST = False", setup.source, flags=re.M), f"{name}: expected SMOKE_TEST = False (full run)"
    nb.cells.insert(0, nbformat.v4.new_code_cell(prelude))
    out = os.path.join(out_dir, name + ".ipynb")
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    t0, done = time.time(), [0]

    def after_cell(cell=None, cell_index=None, **_):
        if cell is not None and cell.cell_type == "code":
            done[0] += 1
        nbformat.write(nb, out)
        err = any(o.get("output_type") == "error" for o in (cell.get("outputs", []) if cell is not None else []))
        print(f"{time.strftime('%H:%M')}  {name}: code cell {done[0]}/{n_code} {'ERROR' if err else 'done'}  "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)

    print(f"▶ FULL RUN {name} started {time.strftime('%Y-%m-%d %H:%M')} — executed copy: {out}", flush=True)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": cwd}},
                            on_cell_executed=after_cell)
    try:
        client.execute()
        ok = True
    except CellExecutionError as e:
        ok = False
        print(str(e)[-8000:])
    nbformat.write(nb, out)
    print(f"{'PASS' if ok else 'FAIL'} FULL {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return ok
