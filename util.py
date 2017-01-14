import re


class UnsupportedFeature(Exception):
    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return '`%s` are not supported in this version of the FRISC-C compiler.' % self.feature


class CompileError(Exception):
    def __init__(self, msg, node):
        self.msg = msg
        self.node = node

    def __str__(self):
        return 'Compiler error: `%s` [%s]' % (self.msg, self.node.coord)


class Scope:
    """Nested scope object containing variables"""

    def __init__(self, parent=None):
        self.items = {}
        self.parent = parent
        self.depth = parent.depth + 1 if parent is not None else 0
        self.space = 0

    def __str__(self):
        return '<Scope (#%d): %s> [%d]' % (self.depth, self.items, self.space)

    def __contains__(self, key):
        return key in self.items or key in self.parent

    def __getitem__(self, key):
        value = (self.items.get(key, None) or
                 (self.parent[key] if self.parent is not None else None))
        if value is None:
            raise KeyError('Unknown identifier: `%s`' % key)
        return value

    def __setitem__(self, key, value):
        self.items[key] = value

    def register_space(self, size):
        if self.depth > 1:
            self.parent.register_space(size)
        else:
            self.space += size

    def registered_space(self):
        return self.parent.registered_space() if self.depth > 1 else self.space


LOAD_SIZE_MODIFIER = {1: 'B', 2: 'H', 4: ''}
LOG_TYPE_SIZES = {1: 0, 2: 1, 4: 2}


class Type:
    """Object base type"""

    def __repr__(self):
        return str(self)

    def size():
        raise NotImplementedError


class Primitive(Type):
    """Primitive types: int, char, short, void"""
    TYPE_SIZES = {'int': 4, 'short': 2, 'char': 1, 'void': 0}
    TYPE_DEFINES = {'int': 'DW', 'short': 'DH', 'char': 'DB'}

    def __init__(self, typename, unsigned=False):
        self.typename = typename
        self.unsigned = unsigned

    def __str__(self):
        return ('UNSIGNED ' if self.unsigned else '') + self.typename.upper()

    def size(self):
        return self.TYPE_SIZES[self.typename]

    def define_value(self, value):
        if self.typename == 'void':
            raise TypeError('VOID can have no value!')
        value = parse_int(value) if self.typename != 'char' else ord(value[1])
        return '%s 0%X' % (self.TYPE_DEFINES[self.typename], value)


class Array(Type):
    def __init__(self, type, length=-1):
        self.type = type
        self.length = length

    def __str__(self):
        return '%s[%s]' % (self.type, self.length if self.length != -1 else '')

    def size(self):
        return self.length * self.type.size() if self.length != -1 else 4  # a pointer

    def elem_size(self):
        return self.type.size()

    def define_value(self, value):
        if self.type.typename == 'char':
            value = map(ord, value.strip('"') + '\0')
        return '%s %s' % (Primitive.TYPE_DEFINES[self.type.typename], ', '.join(map(lambda x: '0%X' % x, value)))


class Pointer(Type):
    def __init__(self, type):
        self.type = type

    def __str__(self):
        return '%s*' % self.type

    def size(self):
        return 4


class FunctionType(Type):
    def __init__(self, return_type, param_types=[Primitive('void')], params=[]):
        self.return_type = return_type
        self.param_types = param_types
        self._params = params

    def __str__(self):
        return '%s -> %s' % (self.param_types, self.return_type)

    def size(self):
        return 0


class Value:
    """A value of an expression"""

    def __repr__(self):
        return str(self)

    def bool_evaluate(self):
        code = self.fetch_to_stack()
        code += ['; Bool evaluation',
                 '\tPOP R0',
                 '\tCMP R0, 0',
                 '\tMOVE SR, R0',
                 '\tAND R0, 8, R0',
                 '\tXOR R0, 8, R0',
                 '\tSHR R0, 3, R0',
                 '\tPUSH R0']
        return code

    def fetch_to_stack(self):
        raise NotImplementedError

    def save_from_stack(self):
        raise NotImplementedError

    def fetch_address_to_stack(self):
        raise NotImplementedError


class Variable(Value):

    def __init__(self, name, type, address):
        self.name = name
        self.type = type
        self.size = self.type.size()
        self.address = address

    def __str__(self):
        return '<%s var> %s @ [%s]' % (self.type, self.name, self.address)

    def fetch_to_stack(self):
        code = ['; Fetching variable %s' % self]
        if isinstance(self.type, (Primitive, Pointer)):
            code += ['\tLOAD%s R4, (%s)' % (LOAD_SIZE_MODIFIER[self.size], self.address),
                     '\tPUSH R4']
        elif isinstance(self.type, Array):
            code += fetch_address(self.address)
        return code

    def save_from_stack(self):
        if isinstance(self.type, (Primitive, Pointer)):
            return ['; Storing variable %s' % self,
                    '\tPOP R4',
                    '\tSTORE%s R4, (%s)' % (LOAD_SIZE_MODIFIER[self.size], self.address)]
        elif isinstance(self.type, Array):
            print('Storing array -> sz = %d, ln = %d' % (self.type.size(), self.type.length))
            code = ['; Storing variable %s (array)' % self]
            code += fetch_address(self.address)
            code += ['\tPOP R4']
            code += ['\tPOP R0',
                     '\tSTORE%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.type.type.size()],
                     '\tADD R4, 0%X, R4' % self.type.type.size()] * self.type.length
            return code

    def fetch_address_to_stack(self):
        if isinstance(self.type, (Primitive, Pointer)):
            return fetch_address(self.address)


class Constant(Value):

    def __init__(self, type, value, address):
        self.type = type
        self.size = self.type.size()
        self.value = value
        self.address = address

    def __str__(self):
        return '<%s const> %s @ [%s]' % (self.type, self.value, self.address)

    def fetch_to_stack(self):
        if isinstance(self.type, Primitive):
            return ['; Fetching constant %s' % self.address,
                    '\tLOAD%s R4, (%s)' % (LOAD_SIZE_MODIFIER[self.size], self.address),
                    '\tPUSH R4']
        elif isinstance(self.type, Array):
            code = ['; Fetching constant array',
                    '\tMOVE %s, R4' % self.address,
                    '\tMOVE 0%X, R1' % self.size,
                    '\tADD R4, R1, R4']
            code += ['\tSUB R4, 0%X, R4' % self.type.type.size(),
                     '\tLOAD%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.type.type.size()],
                     '\tPUSH R0'] * self.type.length
            return code


class Intermediate(Value):

    def __init__(self, type, location):
        self.type = type
        self.location = location

    def __str__(self):
        return '%s @ %s' % (self.type, self.location)

    def size(self):
        return self.type.size()

    def fetch_to_stack(self):
        return []


class Bool(Value):
    def __init__(self):
        pass

    def __str__(self):
        return 'BOOL @ S'

    def fetch_to_stack(self):
        return []

    def bool_evaluate(self):
        return []


class InitListValue(Value):
    def __init__(self, length):
        self.length = length

    def __str__(self):
        return 'INITLIST @ S'

    def fetch_to_stack(self):
        return []


class ArrayReference(Value):

    def __init__(self, base_code, base_value, index_code, index_value):
        self.base_code, self.base_value = base_code, base_value
        self.index_code, self.index_value = index_code, index_value
        self.type = self.base_value.type if isinstance(self.base_value, Array) else Primitive('int')

    def __str__(self):
        return '<array_ref %s [%s]>' % (self.base_value, self.index_value)

    def fetch_to_stack(self):
        return (['; Array access fetch'] +
                self.index_code + self.index_value.fetch_to_stack() +
                self.base_code + self.base_value.fetch_to_stack() +
                ['\tPOP R0',
                 '\tPOP R1',
                 '\tMOVE 0%X, R2' % self.base_value.type.elem_size(),
                 '\tCALL _MUL',
                 '\tADD R6, R0, R4',
                 '\tLOAD%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.base_value.type.elem_size()],
                 '\tPUSH R0'])

    def save_from_stack(self):
        return (['; Array access save'] +
                self.index_code + self.index_value.fetch_to_stack() +
                self.base_code + self.base_value.fetch_to_stack() +
                ['\tPOP R0',
                 '\tPOP R1',
                 '\tMOVE 0%X, R2' % self.base_value.type.elem_size(),
                 '\tCALL _MUL',
                 '\tADD R6, R0, R4',
                 '\tPOP R0',
                 '\tSTORE%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.base_value.type.elem_size()]])


class PointerReference(Value):
    def __init__(self, ptr_code, ptr_value):
        self.ptr_code = ptr_code
        self.ptr_value = ptr_value
        self.type = ptr_value.type.type

    def __str__(self):
        return '<ptr_ref %s>' % self.ptr_value

    def fetch_to_stack(self):
        return (['; Fetching pointer value %s' % self.ptr_value] +
                self.ptr_code + self.ptr_value.fetch_to_stack() +
                ['\tPOP R4',
                 '\tLOAD%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.ptr_value.type.type.size()],
                 '\tPUSH R0'])

    def save_from_stack(self):
        return (['; Storing pointer value %s' % self.ptr_value] +
                self.ptr_code + self.ptr_value.fetch_to_stack() +
                ['\tPOP R4',
                 '\tPOP R0',
                 '\tSTORE%s R0, (R4)' % LOAD_SIZE_MODIFIER[self.ptr_value.type.type.size()]])


class Function(Value):
    def __init__(self, name, type, params, address):
        self.name = name
        self.type = type
        self.params = params
        self.address = address

    def __str__(self):
        return '<fn %s :: %s> (%s>' % (self.name, self.type, self.params)

    def params_size(self):
        return sum(param.size() for param in self.params)


def parse_int(s):
    if s.startswith('0x'):
        return int(s, 16)
    elif s.startswith('0b'):
        return int(s, 2)
    elif s.startswith('0'):
        return int(s, 8)
    else:
        return int(s)


def fetch_address(a):
    parts = re.split('(\+|\-)', a)
    if len(parts) > 1:  # register-based address
        return ['\t%s %s, 0%X, R0' % ('ADD' if parts[1] == '+' else 'SUB', parts[0], int(parts[2], 16)),
                '\tPUSH R0']
    else:               # label-based address
        return ['\tMOVE %s, R0' % a,
                '\tPUSH R0']


def supertype(type_a, type_b):
    if isinstance(type_a, Pointer):
        type_s = type_a
    elif isinstance(type_a, Array):
        type_s = type_a
    else:
        type_s = max(type_a, type_b, key=lambda x: Primitive.TYPE_SIZES[x.typename])
        type_s.unsigned = type_a.unsigned and type_b.unsigned
    return type_s
