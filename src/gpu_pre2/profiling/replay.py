"""Read a registered I2/I3 capture; write a separate audit, never alter raw."""
import argparse
import csv
from pathlib import Path
from gpu_pre2.data.storage import REPO, data_root, inside_data
from gpu_pre2.i2.common import read
from .audit_v5 import audit
from .protocols import validate_request

def capture(stage, index):
    if stage not in {'I2','I3'}:
        raise ValueError('Only retained I2/I3 capture indices are accepted')
    with (REPO/f'data/manifests/{stage}_CELLS.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not 0 <= index < len(rows):raise ValueError('Cell index out of range')
    row=rows[index]
    key=next(k for k in ('path','cell_path','root') if k in row)
    cell=inside_data(data_root()/row[key])
    if row['mode']=='GI':raise ValueError('GI is not a profiling trace')
    validate_request(read(cell/'request.json'))
    return cell

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['I2','I3'],required=True)
    parser.add_argument('--cell-index',type=int,required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if out==data_root() or data_root() in out.parents:
        raise ValueError('Audit output must be outside the immutable data root')
    print(audit(capture(args.stage,args.cell_index),out))

if __name__=='__main__':main()
