from pycparser.c_ast import *

from util import *

ALU_OPERATIONS = {'+': 'ADD', '-': 'SUB', '>>': 'ASHR', '<<': 'SHL', '|': 'OR',
                  '&': 'AND', '^': 'XOR', '*': None, '/': None, '%': None}
EQ_OPERATIONS = {'==': 'EQ', '!=': 'NE'}
CMP_OPERATIONS = {'>': 'GT', '<': 'LT', '>=': 'GE', '<=': 'LE'}
LOGICAL_OPERATIONS = {'&&': 'AND', '||': 'OR', '^^': 'XOR'}
SPECIAL_ALU_OPERATIONS = {'*': '_MUL', '/': '_DIV', '%': '_MOD'}


class FriscGenerator(NodeVisitor):
    INIT_SP = 0x1000

    def __init__(self):
        self.generic_labels = {'fn': 0, 'fn_ret': 0,
                               'var': 0, 'const': 0,
                               'if_true': 0, 'if_false': 0,
                               'cmp_true': 0, 'cmp_end': 0,
                               'for': 0, 'for_end': 0,
                               'while': 0, 'while_end': 0,
                               'dowhile': 0, 'dowhile_end': 0, 'dowhile_cond': 0}
        self.constants = {}
        self.labels = {}
        self.memory = {'data': [''], 'header': [], 'code': []}
        self.registers = {'R0': None, 'R1': None, 'R2': None, 'R3': None}

        # Main function location (label)
        self.main_fn = None

        # Is there multiplication, division or modulo operations?
        self.has_MDM = True

    def generate(self, ast):
        ast.show()
        self.visit(ast, Scope())
        self._generate_header()
        print('=' * 80)
        print('\n'.join(self.memory['header'] + self.memory['code'] + self.memory['data']))
        print('=' * 80)

    def visit(self, node, scope=None, labels=None):
        """Return results of visiting a node with a proper method"""
        visitor = getattr(self, 'visit_' + node.__class__.__name__,
                          self.generic_visit)
        return visitor(node, scope, labels)

    def generic_visit(self, node, scope, labels=None):
        print('Generic visit of', node)
        for c_name, c in node.children():
            self.visit(c, scope)

    # Node visitors

    def visit_ArrayDecl(self, node, scope, labels=None) -> Type:
        length = -1
        if node.dim is not None:
            _, dim = self.visit(node.dim, scope, labels)
            if not isinstance(dim, Constant):
                raise CompileError('Cannot have array of non-constant size!', node)
            length = parse_int(dim.value)
        return Array(self.visit(node.type, scope, labels), length)

    def visit_ArrayRef(self, node, scope, labels=None) -> ([], ArrayReference):
        base_code, base_value = self.visit(node.name, scope, labels)
        index_code, index_value = self.visit(node.subscript, scope, labels)
        ar = ArrayReference(base_code, base_value, index_code, index_value)
        return [], ar

    def visit_Assignment(self, node, scope, labels=None) -> [str]:
        if node.op != '=':
            node.rvalue = BinaryOp(node.op[:-1], node.lvalue, node.rvalue)
        code = ['; Assignment @%s' % node.coord]
        rcode, rvalue = self.visit(node.rvalue, scope, labels)
        lcode, lvalue = self.visit(node.lvalue, scope, labels)
        code += rcode + rvalue.fetch_to_stack() + lcode + lvalue.save_from_stack() + ['']
        return code

    def visit_BinaryOp(self, node, scope, labels=None) -> ([str], Value):
        lcode, lvalue = self.visit(node.left, scope, labels)
        rcode, rvalue = self.visit(node.right, scope, labels)
        if node.op in ALU_OPERATIONS:
            code = ['; ALU op %s' % node.op]
            code += rcode + rvalue.fetch_to_stack() + lcode + lvalue.fetch_to_stack()
            if node.op in SPECIAL_ALU_OPERATIONS:
                code += ['\tCALL %s' % SPECIAL_ALU_OPERATIONS[node.op],
                         '\tPUSH R6']
            else:
                code += ['\tPOP R1',
                         '\tPOP R2']
                if node.op in ('+', '-') and isinstance(lvalue.type, Pointer) and not isinstance(rvalue.type, Pointer):
                    code += ['; Pointer arithmetic!',
                             '\tSHL R2, %d, R2' % LOG_TYPE_SIZES[lvalue.type.type.size()]]
                code += ['\t%s R1, R2, R0' % ALU_OPERATIONS[node.op],
                         '\tPUSH R0']
            return code, Intermediate(supertype(lvalue.type, rvalue.type), 'S')
        elif node.op in CMP_OPERATIONS or node.op in EQ_OPERATIONS:  # comparison operations
            cmp_label, cmp_end_label = self._make_label('cmp_true'), self._make_label('cmp_end')
            code = ['; CMP op %s' % node.op]
            code += rcode + rvalue.fetch_to_stack() + lcode + lvalue.fetch_to_stack()
            if node.op in CMP_OPERATIONS:
                stype = supertype(lvalue.type, rvalue.type)
                cond = ('U' if stype.unsigned else 'S') + CMP_OPERATIONS[node.op]
            else:
                cond = EQ_OPERATIONS[node.op]
            code += ['\tPOP R1',
                     '\tPOP R2',
                     '\tCMP R1, R2',
                     '\tJR_%s %s' % (cond, cmp_label),
                     '\tMOVE 0, R0',
                     '\tJR %s' % cmp_end_label,
                     cmp_label,
                     '\tMOVE 1, R0',
                     cmp_end_label,
                     '\tPUSH R0']
            return code, Bool()
        else:  # logical operations
            code = ['; LOGICAL op %s' % node.op]
            code += rcode + rvalue.bool_evaluate() + lcode + lvalue.bool_evaluate()
            code += ['\tPOP R1',
                     '\tPOP R2',
                     '\t%s R1, R2, R0' % LOGICAL_OPERATIONS[node.op],
                     '\tPUSH R0']
            return code, Bool()

    def visit_Break(self, node, scope, labels=None):
        return ['\tJR %s' % labels['end']]

    def visit_Cast(self, node, scope, labels=None):
        ntype = self.visit(node.to_type.type)
        code, value = self.visit(node.expr)
        return code + value.fetch_to_stack(), Intermediate(ntype, 'S')

    def visit_Compound(self, node, scope, labels=None) -> [str]:
        code = []
        local_scope = Scope(scope)
        for _, c in node.children():
            if isinstance(c, Decl):
                dcode, dname, dtype, rvalue = self.visit(c, local_scope, labels)
                if isinstance(dtype, Array) and dtype.length == -1:
                    if isinstance(rvalue, InitListValue):
                        dtype.length = rvalue.length
                    elif isinstance(rvalue.type, Array):
                        dtype.length = rvalue.type.length
                scope[dname] = Variable(dname, dtype, 'R5-0%X' % local_scope.registered_space())
                scope.register_space(dtype.size())
                code += dcode
            else:
                tret = self.visit(c, local_scope, labels)
                if isinstance(tret, tuple):
                    tcode = tret[0]
                    if isinstance(tret[1], (Intermediate, Bool)):
                        tcode += ['\tPOP R0  ; stack cleanup']
                else:
                    tcode = tret
                code += tcode
        return code

    def visit_Constant(self, node, scope, labels=None) -> ([], Constant):
        const = self._make_constant(node.value, node.type)
        return [], const

    def visit_Continue(self, node, scope, labels=None) -> [str]:
        return ['\tJR %s' % labels['continue']]

    def visit_Decl(self, node, scope=None, labels=None) -> ([str], str, Type, Value):
        type = self.visit(node.type, scope, labels)
        code = ['; Declaration of %s @%s' % (node.name, node.coord)]
        if node.init is not None:
            rcode, rvalue = self.visit(node.init, scope, labels)
            code += rcode + rvalue.fetch_to_stack()
        else:
            rvalue = None
        return code, node.name, type, rvalue

    def visit_DeclList(self, node, scope, labels=None):
        raise CompileError('A DeclList! Haven\'t seen this in a while... actually, never! What to do??!', node.coord)

    def visit_DoWhile(self, node, scope, labels=None):
        dowhile_label, dowhile_end_label, dowhile_cond_label = \
            self._make_label('dowhile'), self._make_label('dowhile_end'), self._make_label('dowhile_cond')
        labels.update({'continue': dowhile_cond_label, 'end': dowhile_end_label})
        code = ['; Do-While loop',
                dowhile_label]
        body_ret = self.visit(node.stmt, scope, labels)
        if isinstance(body_ret, tuple):
            body_code, body_value = body_ret
            if isinstance(body_value, (Intermediate, Bool)):
                body_code += ['\tPOP R0  ; stack cleanup']
        else:
            body_code = body_ret
        code += body_code
        cond_code, cond_value = self.visit(node.cond, scope, labels)
        code += [dowhile_cond_label] + cond_code + cond_value.bool_evaluate()
        code += ['\tPOP R0',
                 '\tCMP R0, 0',
                 '\tJR_NE %s' % dowhile_label,
                 dowhile_end_label]
        return code

    def visit_EmptyStatement(self, node, scope, labels=None) -> []:
        return []

    def visit_ExprList(self, node, scope, labels=None) -> [str]:
        code = []
        for e in reversed(node.exprs):
            tcode, tvalue = self.visit(e)
            code += tcode + tvalue.fetch_to_stack()
        return code

    def visit_FileAST(self, node, scope, labels=None) -> None:
        for e in node.ext:
            if isinstance(e, Decl):
                code, dname, dtype, rvalue = self.visit(e, scope, labels)
                if isinstance(dtype, Array) and dtype.length == -1:
                    if isinstance(rvalue, InitListValue):
                        dtype.length = rvalue.length
                    elif isinstance(rvalue.type, Array):
                        dtype.length = rvalue.type.length
                location = self._make_variable(dname, dtype.size())
                scope[dname] = Variable(dname, dtype, location)
                if rvalue is not None:
                    code += scope[dname].save_from_stack()
                self.memory['code'] += code
            else:
                self.visit(e, scope, labels)

    # def visit_For(self, node, scope, labels=None):
    #     pass
    # for (init; cond; next) stmt
    #
    # For: [init*, cond*, next*, stmt*]

    def visit_FuncCall(self, node, scope, labels=None):
        _, func = self.visit(node.name, scope, labels)
        code = ['; Function call %s' % func.name]
        code += self.visit(node.args, scope, labels)
        code += ['\tCALL %s' % func.address,
                 '\tPUSH R6']
        return code, Intermediate(Primitive('int'), 'S')

    def visit_FuncDecl(self, node, scope=None, labels=None) -> (FunctionType, []):
        params = self.visit(node.args, scope, labels) if node.args is not None else []
        return FunctionType(self.visit(node.type, scope, labels),
                            [param[1] for param in params], params)

    def visit_FuncDef(self, node, scope, labels=None) -> None:
        local_scope = Scope(scope)
        _, name, func_type, _ = self.visit(node.decl, scope, labels)
        params = func_type._params
        start_label, ret_label = self._make_label('fn', name), self._make_label('fn_ret', name)
        scope[name] = Function(name, func_type, params, start_label)

        if name == 'main':
            self.main_fn = start_label

        params_addr = 12
        for param in params:
            local_scope[param[0]] = Variable(param[0], param[1], 'R5+0%X' % params_addr)
            params_addr += 4
        bcode = self.visit(node.body, local_scope, {'fn': start_label, 'fn_ret': ret_label})
        code = (['',
                 start_label,
                 '\tPUSH R5',
                 '\tMOVE SP, R5',
                 '\tSUB SP, 0%X, SP' % local_scope.registered_space()] +
                bcode +
                [ret_label,
                 '\tADD SP, 0%X, SP' % local_scope.registered_space(),
                 '\tPOP R5',
                 '\tRET'])
        self._make_function(name, code)

    def visit_ID(self, node, scope, labels=None) -> ([], Value):
        return [], scope[node.name]

    def visit_IdentifierType(self, node, scope=None, labels=None) -> Primitive:
        return Primitive(node.names[-1], unsigned=node.names[0] == 'unsigned')

    def visit_If(self, node, scope, labels=None) -> [str]:
        if_true_label, if_false_label = self._make_label('if_true'), self._make_label('if_false')
        cond_code, cond_value = self.visit(node.cond, scope, labels)
        true_ret = self.visit(node.iftrue, scope, labels)
        if isinstance(true_ret, tuple):
            true_code, true_value = true_ret
            if isinstance(true_value, (Intermediate, Bool)):
                true_code += ['\tPOP R0  ; stack cleanup']
        else:
            true_code = true_ret
        if node.iffalse:
            false_ret = self.visit(node.iffalse, scope, labels)
            if isinstance(false_ret, tuple):
                false_code, false_value = false_ret
                if isinstance(false_value, (Intermediate, Bool)):
                    false_code += ['\tPOP R0  ; stack cleanup']
            else:
                false_code = false_ret
        else:
            false_code = []
        code = ['; If test'] + cond_code + cond_value.bool_evaluate()
        code += ['\tPOP R0',
                 '\tCMP R0, 0',
                 '\tJR_EQ %s' % if_false_label,
                 if_true_label]
        code += true_code + [if_false_label] + false_code
        return code

    def visit_InitList(self, node, scope, labels=None):
        code = ['; Init list']
        for expr in reversed(node.exprs):
            tcode, tvalue = self.visit(expr, scope, labels)
            code += tcode + tvalue.fetch_to_stack()
        return code, InitListValue(len(node.exprs))

    def visit_ParamList(self, node, scope=None, labels=None) -> [(str, Type)]:
        return [self.visit(param, scope, labels)[1:] for param in node.params]

    def visit_PtrDecl(self, node, scope=None, labels=None) -> Pointer:
        return Pointer(self.visit(node.type, scope, labels))

    def visit_Return(self, node, scope, labels=None) -> [str]:
        code, value = self.visit(node.expr, scope, labels)
        return code + value.fetch_to_stack() + ['\tPOP R6', '\tJR %s' % labels['fn_ret']]

    def visit_TypeDecl(self, node, scope=None, labels=None):
        return self.visit(node.type, scope, labels)

    def visit_UnaryOp(self, node, scope, labels=None) -> ([str], Value):
        code, value = self.visit(node.expr, scope, labels)
        code = ['; Unary operator %s' % node.op] + code
        if node.op == '&':
            code += value.fetch_address_to_stack()
            value = Intermediate(Primitive('int'), 'S')
        elif node.op == '*':
            if not isinstance(value.type, Pointer):
                raise CompileError('Not a pointer!', node)
            code, value = [], PointerReference(code, value)
        elif node.op == '+':
            pass
        elif node.op == '-':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tMOVE 0, R0',
                     '\tSUB R0, R1, R0',
                     '\tPUSH R0']
            value = Intermediate(value.type, 'S')
        elif node.op == '~':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tXOR R1, -1, R1',
                     '\tPUSH R1']
            value = Intermediate(value.type, 'S')
        elif node.op == '!':
            code += value.bool_evaluate()
            code += ['\tPOP R1',
                     '\tXOR R1, 1, R1',
                     '\tPUSH R1']
        elif node.op == '++':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tADD R1, %d, R1' % (1 if not isinstance(value.type, Pointer) else LOG_TYPE_SIZES[value.type.type.size]),
                     '\tPUSH R1',
                     '\tPUSH R1']
            code += value.save_from_stack()
            value = Intermediate(value.type, 'S')
        elif node.op == 'p++':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tPUSH R1',
                     '\tADD R1, %d, R1' % (1 if not isinstance(value.type, Pointer) else LOG_TYPE_SIZES[value.type.type.size]),
                     '\tPUSH R1']
            code += value.save_from_stack()
            value = Intermediate(value.type, 'S')
        elif node.op == '--':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tSUB R1, %d, R1' % (1 if not isinstance(value.type, Pointer) else LOG_TYPE_SIZES[value.type.type.size]),
                     '\tPUSH R1',
                     '\tPUSH R1']
            code += value.save_from_stack()
            value = Intermediate(value.type, 'S')
        elif node.op == 'p--':
            code += value.fetch_to_stack()
            code += ['\tPOP R1',
                     '\tPUSH R1',
                     '\tSUB R1, %d, R1' % (1 if not isinstance(value.type, Pointer) else LOG_TYPE_SIZES[value.type.type.size]),
                     '\tPUSH R1']
            code += value.save_from_stack()
            value = Intermediate(value.type, 'S')

        return code, value

    def visit_While(self, node, scope, labels=None) -> [str]:
        while_label, while_end_label = self._make_label('while'), self._make_label('while_end')
        labels.update({'continue': while_label, 'end': while_end_label})
        cond_code, cond_value = self.visit(node.cond, scope, labels)
        code = ['; While loop',
                while_label]
        code += cond_code + cond_value.bool_evaluate()
        code += ['\tPOP R0',
                 '\tCMP R0, 0',
                 '\tJR_EQ %s' % while_end_label]
        body_ret = self.visit(node.stmt, scope, labels)
        if isinstance(body_ret, tuple):
            body_code, body_value = body_ret
            if isinstance(body_value, (Intermediate, Bool)):
                body_code += ['\tPOP R0  ; stack cleanup']
        else:
            body_code = body_ret
        code += body_code + ['\tJR %s' % while_label, while_end_label]
        return code

    # Unsupported features

    def visit_Case(self, node, scope, labels=None):
        raise UnsupportedFeature('Switch-Cases')

    def visit_CompoundLiteral(self, node, scope, labels=None):
        raise UnsupportedFeature('Compund literals')

    def visit_Default(self, node, scope, labels=None):
        raise UnsupportedFeature('Switch-Cases')

    def visit_EllipsisParam(self, node, scope, labels=None):
        raise UnsupportedFeature('Ellipsis params')

    def visit_Enum(self, node, scope, labels=None):
        raise UnsupportedFeature('Enums')

    def visit_Enumerator(self, node, scope, labels=None):
        raise UnsupportedFeature('Enums')

    def visit_EnumeratorList(self, node, scope, labels=None):
        raise UnsupportedFeature('Enums')

    def visit_Goto(self, node, scope, labels=None) -> [str]:
        raise UnsupportedFeature('Gotos')

    def visit_Label(self, node, scope, labels=None) -> [str]:
        raise UnsupportedFeature('Labels')

    def visit_NamedInitializer(self, node, scope, labels=None):
        raise UnsupportedFeature('Named initializers')

    def visit_Struct(self, node, scope, labels=None):
        raise UnsupportedFeature('Structs')

    def visit_StructRef(self, node, scope, labels=None):
        raise UnsupportedFeature('Structs')

    def visit_Switch(self, node, scope, labels=None):
        raise UnsupportedFeature('Switch-Cases')

    def visit_TernaryOp(self, node, scope, labels=None):
        raise UnsupportedFeature('Ternary operators')
    # cond ? iftrue : iffalse
    #
    # TernaryOp: [cond*, iftrue*, iffalse*]

    def visit_Typedef(self, node, scope, labels=None):
        raise UnsupportedFeature('Typedefs')

    def visit_Union(self, node, scope, labels=None):
        raise UnsupportedFeature('Unions')

    def visit_Pragma(self, node, scope, labels=None):
        raise UnsupportedFeature('Pragmas')

    def _generate_header(self):
        if self.main_fn is None:
            raise Exception('main() function not defined anywhere in the code!!!')
        self.memory['header'] = ['\tORG 0',
                                 '\tMOVE 0%X, SP' % self.INIT_SP,
                                 '\tCALL %s' % self.main_fn,
                                 '\tHALT',
                                 '']
        if self.has_MDM:
            self.memory['header'].extend(['_MUL',
                                          '\tMOVE 0, R6',
                                          '_MULLOOP',
                                          '\tCMP R2, 0',
                                          '\tJP_Z _MULEND',
                                          '\tADD R6, R1, R6',
                                          '\tSUB R2, 1, R2',
                                          '\tJP _MULLOOP',
                                          '_MULEND',
                                          '\tRET',
                                          ''])
            self.memory['header'].extend(['_DIV',
                                          '\tMOVE 0, R6',
                                          '_DIVLOOP',
                                          '\tSUB R1, R2, R1',
                                          '\tJP_ULT _DIVEND',
                                          '\tADD R6, 1, R6',
                                          '\tJP _DIVLOOP',
                                          '_DIVEND',
                                          '\tRET',
                                          ''])
            self.memory['header'].extend(['_MOD',
                                          '\tSUB R1, R2, R1',
                                          '\tJP_ULT _MODEND',
                                          '\tJP _MOD',
                                          '_MODEND',
                                          '\tADD R1, R2, R6',
                                          '\tRET',
                                          ''])

    def _make_label(self, name, spec=''):
        cnt = self.generic_labels[name]
        self.generic_labels[name] += 1
        label = '%s_%d' % (name, cnt)
        if spec:
            label += '_%s' % spec
        return label

    def _make_constant(self, value, type):
        if (value, type) not in self.constants:
            label = self._make_label('const')
            const_type = Primitive(type) if type != 'string' else Array(Primitive('char'), len(value) - 1)
            self.constants[(value, type)] = Constant(const_type, value, label)
            self.memory['data'].append('%s\t%s' % (label, const_type.define_value(value)))
        return self.constants[(value, type)]

    def _make_variable(self, name, size, value=None):
        label = self._make_label('var', name)
        self.memory['data'].append('%s DS 0%X' % (label, size))
        return label

    def _make_function(self, name, code):
        self.memory['code'].extend(code)
