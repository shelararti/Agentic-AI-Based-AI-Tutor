"""Tools the Ask agent can call (the 'Action' step of ReAct)."""

import ast
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def calculator(expression: str) -> str:
    """Evaluate plain arithmetic safely (no names, no calls, no attribute access)."""
    expression = (expression or "").strip().replace("^", "**")
    if not expression or len(expression) > 100:
        return "Error: give a short arithmetic expression like (12*3)/4."

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](a, b)
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    try:
        result = ev(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, OverflowError) as e:
        return f"Error: {e}"
    return str(round(result, 6) if isinstance(result, float) else result)
