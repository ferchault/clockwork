


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version="1.0")
    parser.add_argument('--format', action='store', help='', metavar='FMT')

    args = parser.parse_args()

    parser.add_argument('--sdf', type=str, help='SDF file', metavar='file', default="~/db/qm9s.sdf.gz")

    return

if __name__ == '__main__':
    main()
