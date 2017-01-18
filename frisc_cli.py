import contextlib
import glob
import os
import serial
import sys
import time

from assembler import assemble
from build import convert_to_mem
from compiler import compile
from utils import bin_to_pretty_hex


def split_list(ls):
    ll, tl = [], []
    for e in ls:
        if e[0] == '@':
            ll.append(tl)
            tl = [e]
        else:
            tl.append(e)
    ll.append(tl)
    return ll[1:]


def cprint(*args, color='default', **kwargs):
    if not _STYLE_DISABLE_FLAG:
        print(_ANSI_COLOR_SEQ[color], end='', flush=True)
    print(*args, **kwargs)
    if not _STYLE_DISABLE_FLAG:
        print(_ANSI_COLOR_SEQ['reset'], end='', flush=True)


def connect_to_serial(baudrate, port=None):
    if port is None:
        conn_names = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        conn = serial.Serial(conn_names[0], baudrate=baudrate, timeout=0.2) if conn_names else None
    else:
        conn = serial.Serial(port, baudrate=baudrate, timeout=0.2)
    if conn is not None:
        conn.reset_input_buffer()
        conn.reset_output_buffer()
    return conn


def read_line(ser):
    return ser.readline().decode().rstrip()


def write_hex(ser, i):
    ser.write(('%08X' % i).encode())


def read_hex(ser):
    try:
        return int(read_line(ser), 16)
    except Exception as e:
        print(e)
        return 0


def display_from_serial(ser, suppress_prompt=True):
    time.sleep(0.1)
    while ser.in_waiting > 0:
        line = read_line(ser)
        if suppress_prompt and line == _FRISC_PROMPT:
            continue
        cprint('>', '%s' % line, color='white')


def read_command():
    is_read = False
    while not is_read:
        cprint('FRISC>', end=' ', flush=True, color='cyan')
        inp = input()
        if not inp:
            continue
        cmd, *args = inp.split()
        if cmd in _CLI_COMMANDS:
            is_read = True
        else:
            cprint('! Unknown command `%s`' % cmd, color='red')
    return cmd, args


def reset_processor(ser, *args):
    cprint('* Software-resetting FRISC processor', color='yellow')
    ser.write(b'R')
    display_from_serial(ser)


def load_program(ser, *args):
    if len(args) not in (1, 2, 3):
        cprint('! Wrong command call, should be `load file.a [start_location] [use_symtable]`', color='red')
        return
    file = args[0]
    start_location = int(args[1], 16) if len(args) > 1 else _DEFAULT_PROG_START
    use_symtable = eval(args[2]) if len(args) > 2 else True
    cprint('* Loading assembly file `%s` into FRISC' % file, color='blue')
    filename = file.rsplit('.', maxsplit=1)[0] + '.tmp'
    if use_symtable:
        with open('resources/sym_table.a', 'r') as sym_table:
            syms = sym_table.readlines()
    else:
        syms = []
    with open(file, 'r') as f:
        content = f.readlines()
    with open(filename + '.a', 'w') as f:
        f.writelines(syms + content)

    res = assemble(filename + '.a', start_num=start_location)
    if not res[1]:
        cprint('! Assembling failed\n', ' ', res[0], color='red')
        return
    cprint('* File successfully assembled', color='green')
    convert_to_mem(filename)
    cprint('* File converted to mem format', color='green')
    with open(filename + '.mem', 'r') as f:
        for block in split_list(map(lambda x: x.strip(), f.readlines())):
            address = int(block[0][1:], 16)
            cprint('* Writing to FRISC memory block @%08X(%d)' % (address, len(block) - 1), color='blue')
            ser.write(b'S')
            write_hex(ser, address)
            write_hex(ser, len(block) - 1)
            for value in block[1:]:
                print(value)
                write_hex(ser, int(''.join(reversed(value.split())), 16))
            for _ in range(3):
                display_from_serial(ser)
    os.remove(filename + '.a')
    os.remove(filename + '.p')
    os.remove(filename + '.mem')
    cprint('* Done loading program', color='green')


def run_c_program(ser, *args):
    if len(args) != 1:
        cprint('! Wrong command call, should be `load_c file.c`', color='red')
        return
    file = args[0]
    start_location = _DEFAULT_PROG_START
    cprint('* Compiling C file `%s`' % file, color='blue')
    compile(file)
    afile = file.rsplit('.', maxsplit=1)[0] + '.a'
    cprint('* Loading assembly file `%s` into FRISC' % afile, color='blue')
    filename = afile.rsplit('.', maxsplit=1)[0] + '.tmp'
    with open(afile, 'r') as f:
        content = f.readlines()
    with open(filename + '.a', 'w') as f:
        f.writelines(["RETURN                  EQU   0168\n"] + content)

    res = assemble(filename + '.a', start_num=start_location)
    if not res[1]:
        cprint('! Assembling failed\n', ' ', res[0], color='red')
        return
    cprint('* File successfully assembled', color='green')
    convert_to_mem(filename)
    cprint('* File converted to mem format', color='green')
    with open(filename + '.mem', 'r') as f:
        for block in split_list(map(lambda x: x.strip(), f.readlines())):
            address = int(block[0][1:], 16)
            cprint('* Writing to FRISC memory block @%08X(%d)' % (address, len(block) - 1), color='blue')
            ser.write(b'S')
            write_hex(ser, address)
            write_hex(ser, len(block) - 1)
            for value in block[1:]:
                print(value)
                write_hex(ser, int(''.join(reversed(value.split())), 16))
            for _ in range(3):
                display_from_serial(ser)
    os.remove(afile)
    os.remove(filename + '.a')
    os.remove(filename + '.p')
    os.remove(filename + '.mem')
    cprint('* Done loading program', color='green')
    cprint('* Starting program execution from @%08X' % start_location, color='blue')
    ser.write(b'X')
    write_hex(ser, start_location)
    display_from_serial(ser)


def execute_from(ser, *args):
    if len(args) != 1:
        cprint('! Wrong command call, should be `execute start_location(HEX)`', color='red')
        return
    try:
        location = int(args[0], 16)
        cprint('* Starting program execution from @%08X' % location, color='blue')
        ser.write(b'X')
        write_hex(ser, location)
        display_from_serial(ser)

    except ValueError:
        cprint('! Invalid address: `%s`' % args[0], color='red')


def set_memory(ser, *args):
    if len(args) < 2:
        cprint('! Wrong command call, should be `set_memory start_location(HEX) size(INT) [values(HEX)]`', color='red')
        return
    try:
        start_location = int(args[0], 16)
        size = int(args[1])
        values = map(lambda x: int(x, 16), args[2:])
        cprint('* Writing to FRISC memory block @%08X(%d)' % (start_location, size), color='blue')
        ser.write(b'S')
        write_hex(ser, start_location)
        write_hex(ser, size)
        for v in values:
            write_hex(ser, v)
        display_from_serial(ser)
    except ValueError:
        cprint('! Invalid arguments', color='red')


def get_memory(ser, *args):
    if len(args) != 2:
        cprint('! Wrong command call, should be `get_memory start_location(HEX) size(INT)`', color='red')
        return
    try:
        start_location = int(args[0], 16)
        size = int(args[1])
        cprint('* Reading from FRISC memory block @%08X(%d)' % (start_location, size), color='blue')
        ser.write(b'G')
        write_hex(ser, start_location)
        write_hex(ser, size)
        cprint('>', read_line(ser), color='white')
        cprint('>', read_line(ser), color='white')
        for _ in range(size):
            cprint(' '.join(reversed(bin_to_pretty_hex('{:032b}'.format(read_hex(ser))).split())), color='white')
    except ValueError:
        cprint('! Invalid arguments', color='red')


def dump_memory(ser, *args):
    if len(args) != 3:
        cprint('! Wrong command call, should be `dump_memory start_location(HEX) size(INT) filename`', color='red')
        return
    cprint('* Dumping FRISC memory block contents to `%s`' % args[2], color='blue')
    with contextlib.redirect_stdout(open(args[2], 'w+')):
        global _STYLE_DISABLE_FLAG
        _STYLE_DISABLE_FLAG = True
        get_memory(ser, *args[:2])
        _STYLE_DISABLE_FLAG = False


def halt_processor(ser, *args):
    if args:
        cprint('! Wrong command call, should be `halt`', color='red')
        return
    cprint('* Halting FRISC processor', color='yellow')
    ser.write(b'H')
    display_from_serial(ser)
    sys.exit(0)


def send_msg(ser, *args):
    msg = (' '.join(args) + '\n')
    cprint('* Sending message: %s' % msg, color='blue')
    ser.write(msg.encode())


def receive_msg(ser, *args):
    display_from_serial(ser)


def main(baudrate=9600, port=None):
    print('#' * 80, '#', '#\tFRISC CLI v1.0', '#\tCommand-line interface for programming and managing FRISC processors', '#', '#' * 80, sep='\n')
    ser = connect_to_serial(baudrate, port)
    if ser is None:
        cprint('! Could not find appropriate serial connection port.', color='red')
        sys.exit(1)
    cprint('* FRISC CLI connected to serial port `%s`' % ser.name, color='blue')
    reset_processor(ser)

    while True:
        cmd, args = read_command()
        try:
            _CLI_COMMANDS[cmd](ser, *args)
            display_from_serial(ser)
        except Exception as e:
            print(e)


_ANSI_COLOR_SEQ = {
    'default': '',
    'black': '\x1B[30m',
    'red': '\x1B[31m',
    'green': '\x1B[32m',
    'yellow': '\x1B[33m',
    'blue': '\x1B[34m',
    'magenta': '\x1B[35m',
    'cyan': '\x1B[36m',
    'white': '\x1B[37m',
    'reset': '\x1B[0m'}

_CLI_COMMANDS = {
    'load': load_program,
    'run_c': run_c_program,
    'reset': reset_processor,
    'execute': execute_from,
    'set_memory': set_memory,
    'get_memory': get_memory,
    'dump_memory': dump_memory,
    'halt': halt_processor,
    'send': send_msg,
    'recv': receive_msg,
}

_FRISC_PROMPT = 'FRISC>'
_STYLE_DISABLE_FLAG = True
_DEFAULT_PROG_START = 0x540

if __name__ == '__main__':
    main(*sys.argv[1:])
