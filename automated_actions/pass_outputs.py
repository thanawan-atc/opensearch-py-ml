import json
import sys

def dump(path):
    d = {'folder_path': path}
    with open('output_dict.json', 'w') as f:
        json.dump(d, f, indent=4)

if __name__ == '__main__':
    args = sys.argv[1:]
    dump(args[0])