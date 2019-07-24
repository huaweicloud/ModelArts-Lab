import sys
import os
import argparse
import importlib

__author__ = "Lei Wang (lei.wang1@nio.com)"
__date__ = "28-Aug-2018"
__update__ = "21-March-2019"
__license__ = "MIT"

root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, "%s/core" % root)
sys.path.insert(0, "%s/lib" % root)
sys.path.insert(0, "%s/cloud" % root)
sys.path.insert(0, "%s/config" % root)

def shell(raw_args):
    # you can implement shell selection logics here
    usage = """
cli.py [--<opt>]
    --subShell : enter into a sub shell program
    --prog: execute a program
    """

    parser = argparse.ArgumentParser(description=__doc__, usage=usage)
    parser.add_argument('-s', '--subShell', help="enter into subShell routine")
    parser.add_argument('-e', '--prog', help="execute a program")
    parser.add_argument('argc', nargs=1, type=int)
    parser.add_argument('argv', nargs=argparse.REMAINDER, help="arguments for command")

    args = parser.parse_args(raw_args)

    if args.subShell:
        subShell = None
        return subShell(raw_args)
    elif args.prog:
        [mod_path, method] = args.prog.rsplit('.', maxsplit=1)
        print("mod:", mod_path)
        print("method:", method)
        mod = importlib.import_module(mod_path)
        program = getattr(mod, method, None)
        if program is not None:
            return program(args.argv)
        else:
            print("The program <%s> is invalid!" % program)
    else:
        print("Not valid input!")
        parser.print_help()

if __name__ == "__main__":
    sys.exit(shell(sys.argv[1:]))
