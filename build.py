import os
import platform
import subprocess
import sys

from assembler import assemble


def convert_to_mem(filename):
    last_addr = -1
    lines = []
    with open(filename + '.p', 'r') as inp:
        for line in inp.readlines():
            if not line[:21].strip():
                continue
            addr = int(line[:8], 16) if line[:8].strip() else (last_addr + 4)
            if addr != last_addr + 4:
                lines.append('@%04X' % addr)
            lines.append(line[10:21])
            last_addr = addr

    with open(filename + '.mem', 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Incorrect program call.')
    filename = sys.argv[1].rsplit('.', maxsplit=1)[0]
    print('Assembling source file...')
    res = assemble(filename + '.a', generate_symbol_table=True)
    print('>>', res[0])
    if not res[1]:
        sys.exit(1)
    convert_to_mem(filename)
    print('Converting to mem file.')
    cmd_list = ['data2mem.exe', '-bm', 'resources/frisc_bram.bmm', '-bt',
                'resources/system_wrapper.bit', '-bd', filename + '.mem',
                'tag', 'bram_single_macro_inst', '-o', 'b', filename + '.bit']
    if platform.system() == 'Linux':
        cmd_list.insert(0, 'wine')
    subprocess.call(cmd_list)
    print('Creating bit file.')
    os.remove(filename + '.p')
    os.remove(filename + '.mem')
    print('Removing temp files.')
