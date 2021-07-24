import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)
        
this_dir = osp.dirname(__file__)
print("Path to current directory: {}\n".format(this_dir))

## Add the following to the PYTHONPATH
data_path = osp.join(this_dir, '..')
add_path(data_path)

base_path = osp.abspath(osp.join(this_dir, '..'))
add_path(base_path)

centernet_path = osp.abspath(osp.join(base_path, 'CenterNet', 'src'))
add_path(centernet_path)
