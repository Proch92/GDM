import json


def from_file(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)
