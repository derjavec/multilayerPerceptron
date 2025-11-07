import sys
from utils.get_config import read_config_file

def main():
    if len(sys.argv) > 2:
        raise ValueError('Usage: Please run the program with configuration file, alone or with options')
    if len(sys.argv) == 2:
        read_config_file(sys.argv[1])

if __name__ == '__main__':
    main()