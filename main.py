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
    EOF = 'EOF'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'


# Constants
DIGITS = string.digits
BLANK = string.whitespace
OPERATORS = {
    '+': TT.PLUS,
    '-': TT.MINUS,
    '*': TT.MUL,
    '/': TT.DIV,
    '(': TT.LPAREN,
    ')': TT.RPAREN
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
    def __init__(self, text: str):
        self.pos = 0
        self.text = text
        self.char = text[0]

    def advance(self) -> Lexer:
        self.pos += 1

        if self.pos >= len(self.text):
            self.char = None
            return self

        self.char = self.text[self.pos]
        return self

    def tokenize(self) -> list[Token]:
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

term    : factor ((MUL | DIV) factor)

expr    : term ((PLUS | MINUS) term)
"""


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.token = tokens[self.pos]

    def parse(self):
        return self.expr()

    def advance(self) -> Parser:
        self.pos += 1

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

        elif token.type == TT.LPAREN:
            self.advance()
            expr = self.expr()
            if self.token.type == TT.RPAREN:
                self.advance()
                return expr

    def term(self) -> Node:
        operators = [TT.MUL, TT.DIV]
        return self.bin_op(self.factor, operators)

    def expr(self) -> Node:
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
class Number:
    def __init__(self, value: int | float):
        self.value = value

    def __add__(self, other: Number):
        return Number(self.value + other.value)

    def __sub__(self, other: Number):
        return Number(self.value - other.value)

    def __mul__(self, other: Number):
        return Number(self.value * other.value)

    def __truediv__(self, other: Number):
        return Number(self.value / other.value)

    def __repr__(self):
        return self.value

    def __str__(self):
        return str(self.value)


# Interpreter
class Interpreter:
    def __init__(self, node: Node):
        self.node = node

    def compute(self) -> Number:
        return self.__visit(self.node)

    def __visit(self, node: Node) -> Number:
        if isinstance(node, NumberNode):
            return self.__visitNumberNode(node)
        if isinstance(node, BinOpNode):
            return self.__visitBinOpNode(node)
        if isinstance(node, UnaryOpNode):
            return self.__visitUnaryOpNode(node)

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

        if op.type == TT.PLUS:
            return left + right
        if op.type == TT.MINUS:
            return left - right
        if op.type == TT.DIV:
            return left / right
        if op.type == TT.MUL:
            return left * right


def calc(expression):
    # make tokens
    lexer = Lexer(expression)
    tokens = lexer.tokenize()

    # generate ast
    parser = Parser(tokens)
    ast = parser.parse()

    # get result
    interpreter = Interpreter(ast)
    result = interpreter.compute()

    return result


def main(): 
    while True:
        expression = input('>>> ')
        res = calc(expression)
        print(res.value)


if __name__ == "__main__":
    main()
    
