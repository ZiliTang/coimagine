"""Extract the author's pure candidate function without executing author code.

Supported Python: a candidates list, local scalar expressions, if/else, bounded
range/literal loops, append((option, score)), and one final return candidates.
The resulting guarded expressions are shared by concrete and symbolic checking.
Unrecognized syntax is rejected rather than approximated.
"""
import ast
from copy import deepcopy
import hashlib
import math

from ..symbolic_general.core import Policy, TRUE, evaluate, unit


def Q(value, units):
    """Typed literal in author source; numerically the identity."""
    return value


class ExtractionError(ValueError):
    pass


class Extractor:
    def __init__(self, source, host, derived=None):
        self.host = deepcopy(host)
        self.derived = deepcopy(derived or {})
        self.units = {k: v["unit"] for k, v in host["metrics"].items()}
        for name, expr in self.derived.items():
            if name in self.units:
                raise ExtractionError("derived symbol shadows a primitive")
            self.units[name] = unit(expr, self.units)
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decide"]
        if len(functions) != 1:
            raise ExtractionError("exactly one decide(z) function required")
        self.function = functions[0]
        f = self.function
        if (f.decorator_list or f.args.posonlyargs or f.args.kwonlyargs or
                f.args.defaults or f.args.kw_defaults or f.args.vararg or f.args.kwarg or
                len(f.args.args) != 1 or f.args.args[0].arg != "z" or f.returns or
                f.args.args[0].annotation):
            raise ExtractionError("pure undecorated decide(z) signature required")
        body = list(f.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
        if not body or not (isinstance(body[0], ast.Assign) and
                len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name) and
                body[0].targets[0].id == "candidates" and isinstance(body[0].value, ast.List) and
                not body[0].value.elts):
            raise ExtractionError("start with candidates = []")
        if not (isinstance(body[-1], ast.Return) and isinstance(body[-1].value, ast.Name)
                and body[-1].value.id == "candidates"):
            raise ExtractionError("end with return candidates")
        self.rules = []
        self.nodes = 0
        self.statements(body[1:-1], {}, {}, TRUE)
        if not self.rules:
            raise ExtractionError("function must have at least one candidate emission")
        checked_rules = []
        for r in self.rules:
            checked_rules.append({**r, "when": ["and", r["when"],
                host["options"][r["option"]]["requires"]]})
        self.policy = Policy(dict(version=1, derived=self.derived, rules=checked_rules), host)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.function_sha256 = hashlib.sha256(ast.dump(f).encode()).hexdigest()

    def static(self, node, constants):
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.static(e, constants) for e in node.elts]
        if isinstance(node, ast.JoinedStr):
            pieces = []
            for n in node.values:
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    pieces.append(n.value)
                elif isinstance(n, ast.FormattedValue) and n.conversion == -1 and n.format_spec is None:
                    pieces.append(str(self.static(n.value, constants)))
                else:
                    raise ExtractionError("unsupported formatted key")
            return "".join(pieces)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.static(node.operand, constants)
            if type(value) in (int, float):
                return -value
        raise ExtractionError("static literal or loop variable required")

    def expression(self, node, local, constants):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "z":
            key = self.static(node.slice, constants)
            if key not in self.units:
                raise ExtractionError("unregistered symbol: " + str(key))
            return ["var", key]
        if isinstance(node, ast.Name) and node.id in local:
            return deepcopy(local[node.id])
        if isinstance(node, ast.Constant) or isinstance(node, ast.Name) and node.id in constants:
            v = self.static(node, constants)
            if type(v) is bool:
                return ["lit", v, "bool"]
            if type(v) in (int, float) and math.isfinite(v):
                return ["lit", v, "1"]
            raise ExtractionError("finite scalar expression required")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == "Q" and len(node.args) == 2:
                result = ["lit", self.static(node.args[0], constants), self.static(node.args[1], constants)]
                unit(result, self.units)
                return result
            if name in ("abs", "min", "max") and len(node.args) == (1 if name == "abs" else 2):
                return [name, *[self.expression(a, local, constants) for a in node.args]]
        if isinstance(node, ast.BoolOp):
            return ["and" if isinstance(node.op, ast.And) else "or",
                    *[self.expression(v, local, constants) for v in node.values]]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            return ["not" if isinstance(node.op, ast.Not) else "neg",
                    self.expression(node.operand, local, constants)]
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}.get(type(node.op))
            if op:
                return [op, self.expression(node.left, local, constants),
                        self.expression(node.right, local, constants)]
        if isinstance(node, ast.Compare):
            ops = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
            operands = [node.left, *node.comparators]
            if all(type(op) in ops for op in node.ops):
                parts = [[ops[type(op)], self.expression(a, local, constants),
                          self.expression(b, local, constants)]
                         for op, a, b in zip(node.ops, operands, operands[1:])]
                return parts[0] if len(parts) == 1 else ["and", *parts]
        if isinstance(node, ast.IfExp):
            return ["if", self.expression(node.test, local, constants),
                    self.expression(node.body, local, constants), self.expression(node.orelse, local, constants)]
        raise ExtractionError("unsupported expression: " + ast.dump(node))

    def statements(self, body, local, constants, guard, protected=frozenset()):
        for node in body:
            self.nodes += 1
            if self.nodes > 8192:
                raise ExtractionError("unrolled program exceeds 8192 statements")
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("z", "candidates", "range", "Q", "abs", "min", "max") or name in constants or name in protected:
                    raise ExtractionError("reserved or loop name assignment")
                expr = self.expression(node.value, local, constants)
                unit(expr, self.units)
                local[name] = expr
            elif isinstance(node, ast.If):
                test = self.expression(node.test, local, constants)
                if unit(test, self.units) != "bool":
                    raise ExtractionError("boolean branch condition required")
                # Branch-local assignments may not escape without explicit expressions.
                self.statements(node.body, dict(local), constants, ["and", guard, test], frozenset(local))
                self.statements(node.orelse, dict(local), constants, ["and", guard, ["not", test]], frozenset(local))
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                it = node.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and not it.keywords:
                    args = [self.static(a, constants) for a in it.args]
                    if not 1 <= len(args) <= 3 or any(type(a) is not int for a in args):
                        raise ExtractionError("integer range arguments required")
                    try:
                        values = range(*args)
                    except ValueError as exc:
                        raise ExtractionError("invalid range") from exc
                else:
                    values = self.static(it, constants)
                    if not isinstance(values, list):
                        raise ExtractionError("literal loop sequence required")
                if len(values) > 128 or node.target.id in ("z", "candidates", "range", "Q", "abs", "min", "max") or node.target.id in local or node.target.id in constants:
                    raise ExtractionError("loop limit or target violation")
                for v in values:
                    self.statements(node.body, dict(local), {**constants, node.target.id: v}, guard, frozenset(local))
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "candidates" and node.value.func.attr == "append"
                    and len(node.value.args) == 1 and not node.value.keywords):
                pair = node.value.args[0]
                if not isinstance(pair, ast.Tuple) or len(pair.elts) != 2:
                    raise ExtractionError("emit a tuple (option_name, score)")
                name = self.static(pair.elts[0], constants)
                if name not in self.host["options"]:
                    raise ExtractionError("unknown option: " + str(name))
                score = self.expression(pair.elts[1], local, constants)
                if unit(score, self.units) != "1":
                    raise ExtractionError("scores must be dimensionless")
                self.rules.append(dict(name="emit_" + str(len(self.rules)), option=name, when=guard, score=score))
            else:
                raise ExtractionError("unsupported statement: " + ast.dump(node))

    def candidates(self, primitive, *, startup_only=False):
        values = self.policy.values(primitive)
        rows = []
        for index, rule in enumerate(self.rules):
            if evaluate(rule["when"], values):
                legal = bool(evaluate(self.host["options"][rule["option"]]["requires"], values))
                if legal or not startup_only:
                    rows.append(dict(option=rule["option"], score=evaluate(rule["score"], values),
                                     emission=index, executable=legal))
        return sorted(rows, key=lambda r: (-r["score"], r["emission"]))

    def native_candidates(self, primitive):
        """Independent concrete parity check of the validated original function."""
        namespace = {"__builtins__": {}, "range": range, "Q": Q, "abs": abs, "min": min, "max": max}
        module = ast.Module(body=[self.function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<validated-decide>", "exec"), namespace)
        values = self.policy.values(primitive)
        return namespace["decide"](values)

    def select(self, primitive, admit):
        """Current boundary: first admitted candidate; all scores remain eligible."""
        checks = []
        for row in self.candidates(primitive):
            report = admit(row["option"]) if row["executable"] else {"allowed": False, "reason": "startup"}
            checks.append({**row, "report": report})
            if report["allowed"]:
                return {"selected": row["option"], "checks": checks}
        return {"selected": None, "checks": checks}
