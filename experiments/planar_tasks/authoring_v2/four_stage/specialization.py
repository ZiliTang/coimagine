"""Register exactly the policy's geometric dependencies; retain core skill state."""
from .compiler import Extractor
from .frontend import schema


def specialized_schema(source):
    host=schema()
    program=Extractor(source,host)
    names={k for k in host["metrics"] if ":" not in k or k.startswith("legal:")}
    def visit(expr):
        if not isinstance(expr,list):
            return
        if expr and expr[0]=="var":
            names.add(expr[1])
        else:
            for item in expr[1:]:visit(item)
    for row in program.rules:
        visit(row["when"]);visit(row["score"])
    host["metrics"]={k:v for k,v in host["metrics"].items() if k in names}
    for option in host["options"].values():
        option["writes"]=list(host["metrics"])
    return host
