import sys

from pycparser import parse_file
from generator import FriscGenerator

if __name__ == '__main__':
    ast = parse_file(sys.argv[1], use_cpp=True, cpp_path='clang', cpp_args='-E')
    fg = FriscGenerator()
    fg.generate(ast)

