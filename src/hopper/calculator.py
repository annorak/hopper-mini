import ast
import json
import math
import operator
import os
import subprocess
import sys
from pathlib import Path

_MAX_SOURCE = 8192
_MAX_NODES = 512
_MAX_DEPTH = 32
_MAX_BITS = 1024
_MAX_OUTPUT = 4096
_TIMEOUT = 2
_SANDBOX = "/usr/bin/sandbox-exec"
_PROFILE = "(version 1) (allow default) (deny network*) (deny file-write*) (deny process-fork)"
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_ALLOWED_NODES = {
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Call,
    *_OPERATORS,
}


class CalculatorInvalid(ValueError):
    """The snippet is outside the bounded arithmetic language."""


class CalculatorTimeout(RuntimeError):
    """The calculator exceeded its wall-clock deadline."""


class CalculatorSandboxError(RuntimeError):
    """The isolated worker could not run or failed unexpectedly."""


def _checked(value: int | float) -> int | float:
    if type(value) not in (int, float):
        raise CalculatorInvalid("Only real numbers are supported")
    if isinstance(value, int) and value.bit_length() > _MAX_BITS:
        raise CalculatorInvalid("Integer exceeds the 1024-bit limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorInvalid("Non-finite numbers are not supported")
    return value


def _evaluate(node: ast.expr, variables: dict[str, int | float]) -> int | float:
    match node:
        case ast.Constant(value=value):
            return _checked(value)
        case ast.Name(id=name):
            if name not in variables:
                raise CalculatorInvalid(f"Unknown numeric variable: {name}")
            return variables[name]
        case ast.UnaryOp(op=op, operand=operand):
            value = _evaluate(operand, variables)
            return _checked(-value if isinstance(op, ast.USub) else value)
        case ast.BinOp(left=left, op=op, right=right):
            lhs, rhs = _evaluate(left, variables), _evaluate(right, variables)
            if isinstance(op, ast.Pow) and abs(rhs) > 100:
                raise CalculatorInvalid("Power exponent must be between -100 and 100")
            return _checked(_OPERATORS[type(op)](lhs, rhs))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            if not 1 <= len(args) <= 8:
                raise CalculatorInvalid("Math calls require between 1 and 8 arguments")
            return _checked(_FUNCTIONS[name](*[_evaluate(arg, variables) for arg in args]))
        case _:
            raise CalculatorInvalid("Only numeric expressions and named math calls are allowed")


def _interpret(code: str) -> str:
    tree = ast.parse(code)
    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise CalculatorInvalid("Snippet exceeds the syntax size or depth limit")
        if type(node) not in _ALLOWED_NODES:
            raise CalculatorInvalid(f"Unsupported syntax: {type(node).__name__}")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))

    variables = dict(_CONSTANTS)
    output = []
    output_size = 0
    for statement in tree.body:
        match statement:
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                if name in _FUNCTIONS or name in _CONSTANTS or name == "print":
                    raise CalculatorInvalid(f"Reserved name: {name}")
                variables[name] = _evaluate(value, variables)
                continue
            case ast.Expr(value=ast.Call(func=ast.Name(id="print"), args=args, keywords=[])):
                if len(args) > 8:
                    raise CalculatorInvalid("Print accepts at most 8 numbers")
                line = " ".join(str(_evaluate(arg, variables)) for arg in args)
            case ast.Expr(value=value):
                line = str(_evaluate(value, variables))
            case _:
                raise CalculatorInvalid(
                    "Only single numeric assignments and expressions are allowed"
                )
        output_size += len(line) + 1
        if output_size > _MAX_OUTPUT:
            raise CalculatorInvalid("Output exceeds the 4096-character limit")
        output.append(line)
    return "\n".join(output)


def calculate(code: str) -> str:
    """Run numeric assignments, expressions and print with a two-second limit."""
    if len(code.encode("utf-8")) > _MAX_SOURCE:
        raise CalculatorInvalid("Snippet exceeds the 8192-byte limit")
    if sys.platform != "darwin" or not os.access(_SANDBOX, os.X_OK):
        raise CalculatorSandboxError("Calculator requires macOS sandbox-exec")
    command = [_SANDBOX, "-p", _PROFILE, sys.executable, "-I", "-S", str(Path(__file__).resolve())]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env={"LC_ALL": "C"},
            cwd="/",
            close_fds=True,
        )
    except OSError as error:
        raise CalculatorSandboxError("Could not start calculator sandbox") from error
    try:
        stdout, stderr = process.communicate(code, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise CalculatorTimeout("Calculator exceeded the two-second deadline") from error
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise CalculatorSandboxError(f"Calculator sandbox exited with status {process.returncode}")
    try:
        result = json.loads(stdout)
    except ValueError as error:
        raise CalculatorSandboxError("Calculator worker returned an invalid response") from error
    if result["error"] is not None:
        raise CalculatorInvalid(result["error"])
    return result["output"]


def _worker() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    code = sys.stdin.read(_MAX_SOURCE + 1)
    try:
        if len(code.encode("utf-8")) > _MAX_SOURCE:
            raise CalculatorInvalid("Snippet exceeds the 8192-byte limit")
        result = {"output": _interpret(code), "error": None}
    except (ValueError, TypeError, ArithmeticError, SyntaxError, RecursionError) as error:
        result = {"output": None, "error": str(error)}
    print(json.dumps(result))


if __name__ == "__main__":
    _worker()
