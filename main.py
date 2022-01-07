from __future__ import annotations
from enum import Enum
from typing import Optional, Callable
import string


# Token types
class TT(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MUL = 'MUL'
    DIV = 'DIV'
    REM = 'REM'
    POW = 'POW'
    EOF = 'EOF'
    EQ = 'EQ'
    VAR = 'VAR'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'


# Constants
DIGITS = string.digits
BLANK = string.whitespace
LETTERS = string.ascii_letters
OPERATORS = {
    '+': TT.PLUS,
    '-': TT.MINUS,
    '*': TT.MUL,
    '/': TT.DIV,
    '^': TT.POW,
    '%': TT.REM,
    '(': TT.LPAREN,
    ')': TT.RPAREN,
    '=': TT.EQ
}


# TOKEN
class Token:
    def __init__(self, token_type: TT, token_value: Optional[str | int | float] = None):
        self.type = token_type
        self.value = token_value

    def __str__(self):
        if self.value:
            return f'{self.type.value}:{self.value}'
        return f'{self.type.value}'

    def __repr__(self):
        return self.__str__()


# LEXER
class Lexer:
    def __init__(self):
        self.pos = -1
        self.text = ''
        self.char = ''

    def advance(self) -> Lexer:
        self.pos += 1

        if self.pos >= len(self.text):
            self.char = None
            return self

        self.char = self.text[self.pos]
        return self

    def tokenize(self, expression: str) -> list[Token]:
        self.text = expression
        self.pos = -1
        self.advance()

        tokens = []
        eof = Token(TT.EOF)
        while self.char is not None:
            if self.char in BLANK:
                # Ignore whitespaces and tabs
                self.advance()
            elif self.char in OPERATORS:
                token = Token(OPERATORS[self.char])
                tokens.append(token)
                self.advance()
            elif self.char in DIGITS + '.':
                # make_number advances one more time in the end
                num = self.make_number()

                if isinstance(num, float):
                    token = Token(TT.FLOAT, num)
                else:
                    token = Token(TT.INT, num)

                tokens.append(token)
            elif self.char in LETTERS + '_':
                varname = self.make_varname()
                token = Token(TT.VAR, varname)
                tokens.append(token)

        tokens.append(eof)
        return tokens

    def make_number(self) -> int | float:
        dot = False
        num = ''

        while self.char is not None and self.char in DIGITS + '.':
            if self.char == '.':
                if dot:
                    break
                dot = True

            num += self.char
            self.advance()

        return float(num) if dot else int(num)

    def make_varname(self) -> str:
        varname = ''
        while self.char is not None and self.char in LETTERS + DIGITS + '_':
            varname += self.char
            self.advance()
        return varname


# NODES
class Node:
    def __init__(self, token: Token):
        self.token = token

    def __str__(self):
        return str(self.token)

    def __repr__(self):
        return self.__str__()


class UnaryOpNode(Node):
    def __init__(self, op: Token, node: Node):
        super().__init__(op)
        self.node = node

    def __str__(self):
        return f'({self.token}, {self.node})'


class NumberNode(Node):
    def __init__(self, token: Token):
        super().__init__(token)


class VarNode(Node):
    def __init__(self, token: Token):
        super().__init__(token)


class BinOpNode(Node):
    def __init__(self, left: Node, op: Token, right: Node):
        super().__init__(op)
        self.left = left
        self.right = right

    def __str__(self):
        return f'({self.left}, {self.token}, {self.right})'


# PARSER
"""
Grammar (from highest priority to lowest)

factor  : ((PLUS | MINUS) (INT | FLOAT))
        : INT | FLOAT

term    : factor ((MUL | DIV | POW | REM) factor)

expr    : term ((PLUS | MINUS) term)
        : var EQ expr
"""


class Parser:
    def __init__(self):
        self.tokens = []
        self.pos = -1
        self.token = None

    def parse(self, tokens: list[Token]):
        self.pos = -1
        self.tokens = tokens
        self.advance()

        expr = self.expr()
        if self.token.type is not TT.EOF:
            raise Exception('Invalid syntax')

        return expr

    def advance(self) -> Parser:
        self.pos += 1

        if self.pos >= len(self.tokens):
            self.token = None
            return self

        self.token = self.tokens[self.pos]
        return self

    def goback(self) -> Parser:
        self.pos -= 1

        if self.pos >= len(self.tokens):
            self.token = None
            return self

        self.token = self.tokens[self.pos]
        return self

    def factor(self) -> Node:
        token = self.token

        if token.type in (TT.PLUS, TT.MINUS):
            self.advance()
            factor = self.factor()
            return UnaryOpNode(token, factor)

        elif token.type in (TT.INT, TT.FLOAT):
            self.advance()
            return NumberNode(token)

        elif token.type is TT.LPAREN:
            self.advance()
            expr = self.expr()
            if self.token.type is TT.RPAREN:
                self.advance()
                return expr

        elif token.type == TT.VAR:
            self.advance()
            return VarNode(token)

    def term(self) -> Node:
        operators = [TT.MUL, TT.DIV, TT.POW, TT.REM]
        return self.bin_op(self.factor, operators)

    def expr(self) -> Node:
        if self.token.type is TT.VAR:
            var_token = self.token
            self.advance()
            if self.token.type is TT.EQ:
                eq = self.token
                self.advance()
                expr = self.expr()
                var = VarNode(var_token)
                return BinOpNode(var, eq, expr)
            self.goback()

        operators = [TT.PLUS, TT.MINUS]
        return self.bin_op(self.term, operators)

    def bin_op(
            self,
            fn: Callable[..., Node],
            operators: list[TT]
    ) -> Node:
        left = fn()

        while self.token.type in operators:
            operator = self.token
            self.advance()
            right = fn()

            left = BinOpNode(left, operator, right)

        return left


# Value types
class Value:
    def __init__(self, value: any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: Value):
        return Number(self.value + other.value)

    def __sub__(self, other: Value):
        return Number(self.value - other.value)

    def __mul__(self, other: Value):
        return Number(self.value * other.value)

    def __truediv__(self, other: Value):
        return Number(self.value / other.value)

    def __mod__(self, other: Value):
        return Number(self.value % other.value)

    def __pow__(self, other: Value, modulo=None):
        return Number(self.value ** other.value)


class Variable(Value):
    def __init__(self, name: str):
        super().__init__(None)
        self.name = name

    def set_value(self, value: int | float) -> Variable:
        self.value = value
        return self


class Number(Value):
    def __init__(self, value: int | float):
        super().__init__(value)


# Interpreter
class Interpreter:
    def __init__(self):
        self.variables: dict[str, Variable] = dict()
        self.lexer = Lexer()
        self.parser = Parser()

    def input(self, expression: str) -> any:
        tokens = self.lexer.tokenize(expression)
        ast = self.parser.parse(tokens)
        res = self.__visit(ast)
        return res.value if res else ''

    def __visit(self, node: Node) -> Number | Variable:
        if isinstance(node, NumberNode):
            return self.__visitNumberNode(node)
        if isinstance(node, BinOpNode):
            return self.__visitBinOpNode(node)
        if isinstance(node, UnaryOpNode):
            return self.__visitUnaryOpNode(node)
        if isinstance(node, VarNode):
            return self.__visitVarNode(node)

    def __visitVarNode(self, node: VarNode) -> Variable:
        varname = node.token.value
        return self.variables[varname] if varname in self.variables else Variable(varname)

    def __visitNumberNode(self, node: NumberNode) -> Number:
        return Number(node.token.value)

    def __visitUnaryOpNode(self, node: UnaryOpNode) -> Number:
        num = self.__visit(node.node)
        op = node.token

        if op.type == TT.MINUS:
            num *= Number(-1)

        return num

    def __visitBinOpNode(self, node: BinOpNode) -> Number:
        left = self.__visit(node.left)
        right = self.__visit(node.right)
        op = node.token
        if op.type is TT.PLUS:
            return left + right
        if op.type is TT.MINUS:
            return left - right
        if op.type is TT.DIV:
            return left / right
        if op.type is TT.MUL:
            return left * right
        if op.type is TT.REM:
            return left % right
        if op.type is TT.POW:
            return left ** right
        if op.type is TT.EQ:
            self.variables[left.name] = left.set_value(right.value)
            return right


def main():
    interpreter = Interpreter()
    while True:
        expression = input('>>> ')
        res = interpreter.input(expression)
        print(res)


if __name__ == '__main__':
    main()
