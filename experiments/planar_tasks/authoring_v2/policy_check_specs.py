"""Static evidence declarations for the checked planar CAP author interface."""
import ast

from experiments.planar_tasks.relational_search.motion import ACTIONS


# These are conclusions a bound of this kind could justify. Whether it actually
# does so depends on its value and threshold at the public decision boundary.
WIDTH_USES = {
    'visible_overlap': {'sufficient', 'unknown'},
    'observed_lower_bound': {'sufficient', 'unknown'},
    'certified_upper_bound': {'insufficient', 'unknown'},
    'certified_interval': {'sufficient', 'insufficient', 'unknown'},
}
MODELS = {'stationary', 'co_moving', 'trajectory_set', 'unknown'}


def action_role(code):
    """Public primitive role in the frozen planar option menu."""
    if type(code) is not int or not 0 <= code < len(ACTIONS):
        raise ValueError('invalid public action index')
    if code < 32:
        return 'PUSH'
    if code < 38:
        return 'TURN'
    kind = ACTIONS[code].kind
    return 'HOLD' if kind == 'wait' else kind.upper()


def validate_source_rules(tree):
    """Read the top-level literal without importing or evaluating author code."""
    assignments = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'POLICY_CHECK_RULES'
    ]
    if len(assignments) != 1:
        raise ValueError('declare POLICY_CHECK_RULES exactly once as a top-level literal')
    expression = assignments[0].value
    # A duplicate key in a Python literal would silently erase part of a declaration.
    for node in ast.walk(expression):
        if isinstance(node, ast.Dict):
            if any(not isinstance(k, ast.Constant) or not isinstance(k.value, str) for k in node.keys):
                raise ValueError('POLICY_CHECK_RULES requires literal string keys')
            if len({k.value for k in node.keys}) != len(node.keys):
                raise ValueError('duplicate key in POLICY_CHECK_RULES')
    try:
        rules = ast.literal_eval(expression)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise ValueError('POLICY_CHECK_RULES must be a literal dictionary') from exc
    if not isinstance(rules, dict) or set(rules) != {'version', 'widths', 'peer_models'}:
        raise ValueError('expected version, widths and peer_models in POLICY_CHECK_RULES')
    if type(rules['version']) is not int or rules['version'] != 1:
        raise ValueError('unsupported POLICY_CHECK_RULES version')
    if not isinstance(rules['widths'], dict) or not isinstance(rules['peer_models'], dict):
        raise ValueError('widths and peer_models must be dictionaries')
    for source, uses in rules['widths'].items():
        if source not in WIDTH_USES or not isinstance(uses, list) or not uses:
            raise ValueError(f'unsupported width declaration: {source}')
        if any(not isinstance(u, str) or u not in WIDTH_USES[source] for u in uses):
            raise ValueError(f'unsupported direction of inference from {source}: {uses}')
    roles = {action_role(i) for i in range(len(ACTIONS))}
    for role, options in rules['peer_models'].items():
        if role not in roles or not isinstance(options, list) or not options:
            raise ValueError(f'invalid peer role or model list: {role}')
        for option in options:
            if (not isinstance(option, list) or len(option) != 2
                    or not isinstance(option[0], str) or option[0] not in MODELS
                    or option[1] != 'hypothesis'):
                raise ValueError(f'{role}: physical peer motion requires a declared hypothesis')
    return rules


def check_declared_uses(rules, packet):
    """Match each reported use to the static declaration after packet validation."""
    for row in packet['widths']:
        if row['conclusion'] not in rules['widths'].get(row['source'], []):
            raise ValueError(f"undeclared width use: {row['source']} / {row['conclusion']}")
    for row in packet['peer_predictions']:
        role = action_role(row['pair'][row['robot']])
        if [row['model'], row['scope']] not in rules['peer_models'].get(role, []):
            raise ValueError(f"undeclared peer prediction for {role}: {row['model']} / {row['scope']}")
