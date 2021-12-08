# Methods to count the number of nodes
# in a tree regression

from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)

import jax.numpy as jnp

# Sympy is not safe for parsing.
# https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify

# We'll make a parser based on the example provided here:
#https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py

epsilon = 1e-12

# Operations with infix notation
opn = {
    "+": jnp.add,
    "-": jnp.subtract,
    "*": jnp.multiply,
    "/": jnp.true_divide,
    "^": jnp.power
}

fn = {
    "sin"     : jnp.sin,
    "cos"     : jnp.cos,
    "exp"     : jnp.exp,
    "expn"    : lambda x: jnp.exp(-x),
    "tanh"    : jnp.tanh,
    "arcsin"  : jnp.arcsin,
    "id"      : lambda x: x,
    "log"     : jnp.log,
    "sqrt"    : lambda x: jnp.sqrt(jnp.abs(x)),

    "relu"    : lambda x: jnp.ma.array(x, mask=(x<=0.0), fill_value=0).filled(),
    "gauss"   : lambda x: jnp.exp(-jnp.power(x, 2)),    
    "logit"   : lambda x: jnp.ma.array(jnp.log(x/(1-x)), mask=(x==0.0), fill_value=0).filled(),
    "NOT"     : lambda x: 0.0 if x != 0.0 else 1.0,
    "if"      : lambda x, y, z: y if x!= 0.0 else z,
    "float"   : float,
}

exprStack = []


def push_first(toks):
    exprStack.append(toks[0])

    
def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break

bnf = None

def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        
        # fnumber = Combine(Word("+-"+nums, nums) +
        #                    Optional("." + Optional(Word(nums))) +
        #                    Optional(e + Word("+-"+nums, nums)))
        # or use provided pyparsing_common.number, but convert back to str:
        # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
        
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        # Basic Operations and logical comparators 
        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        les, leq, ge, geq, xor, eq, or_, and_ = Literal('<'), Literal('leq'), Literal('>'), Literal('geq'), Literal('XOR'), Literal('eq'), Literal('OR'), Literal('AND')
        
        # grouping the operators
        addop  = plus | minus
        multop = mult | div
        expop  = Literal("^")
        logic = les | leq | ge | geq | xor | eq | or_ | and_

        #Forward declaration of an expression to be
        #defined later - used for recursive grammars, such as algebraic infix notation.
        expr_logic = Forward()
        factor     = Forward()
        expr       = Forward()
        
        expr_list = delimitedList(Group(expr_logic))
        
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            logic[...]
            + (
                (fn_call | pi | e | fnumber | ident).setParseAction(push_first)
                | Group(lpar + expr_logic + rpar)
            )
        ).setParseAction(push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        
        # precedence: power
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        
        # precedence: multiplication, sum
        term = factor + (multop + factor).setParseAction(push_first)[...]
        
        expr <<= term + (addop + term).setParseAction(push_first)[...]
        
        # precedence: logical operations
        expr_logic <<= expr + (logic + expr).setParseAction(push_first)[...]
        
        bnf = expr_logic
        
    return bnf


def count_nodes(s):
    op, num_args = s.pop(), 0
        
    if isinstance(op, tuple):
        op, num_args = op
    
    if op == "unary -":
        return 1 + count_nodes(s)
    
    if op in [r'+', r'-', r'*', r'/', r'^', r'<', r'leq', r'>', r'geq', r'XOR', 'eq', 'OR', 'AND']:
        # note: operands are pushed onto the stack in reverse order
        left_branch = count_nodes(s)
        right_branch = count_nodes(s)

        return left_branch + right_branch + 1 # opn[op](op1, op2)
        
    elif op == "PI" or op == "E":
        return 1
    
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([count_nodes(s) for _ in range(num_args)])
        return sum(args) + 1
        
    elif "x_" == op[:2]:
        
        # Handling the occurence of variables by
        # getting the index of the variable and
        # returning the value from the given sample xs
        idx = int(op.replace('x_', ''))-1
        
        return 1
    
    elif op[0].isalpha():
        raise Exception("invalid identifier '%s'" % op)
    
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return 1
        
        except ValueError:
            return 1
    
        
def count_expression_nodes(text_representation):
    exprStack[:] = []
    
    results = BNF().parseString(text_representation, parseAll=True)

    return count_nodes(exprStack[:])