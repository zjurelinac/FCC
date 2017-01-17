import platform
import sys

from pycparser import parse_file
from generator import FriscGenerator


def compile(file):
    if platform.system() == 'Windows':
        ast = parse_file(file, use_cpp=True, cpp_path='clang', cpp_args='-E')
    elif platform.system() == 'Linux':
        ast = parse_file(file, use_cpp=True)
    fg = FriscGenerator()
    output = fg.generate(ast)
    outfile = open(file.rsplit('.', maxsplit=1)[0] + '.a', 'w')
    outfile.write(output)
    outfile.close()


if __name__ == '__main__':
    compile(sys.argv[1])
