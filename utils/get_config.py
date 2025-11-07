import argparse
import json
import os

def read_config_file(file_path):
    ext = os.path.splitext(file_path)
    print(ext)