import json
import sys

def dump(w_file, d_path):
    d = {'folder_path': d_path}
    with open(w_file, 'w') as f:
        json.dump(d, f, indent=4)

if __name__ == '__main__':
    args = sys.argv[1:]
    dump(args[0], args[1])