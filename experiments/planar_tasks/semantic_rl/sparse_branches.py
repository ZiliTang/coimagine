"""Select distinct requested commands adjacent to the executed expert in a comparison graph."""


def alternatives(proposals, chosen, neighbors):
    """Group aliases of the same command; timings and target bindings are part of the key."""
    selected=proposals[chosen]
    def key(cmd):
        return (cmd['target_id'],cmd['a0'],cmd['a1'],cmd['ticks'])
    groups={}
    current=key(selected)
    for expert in neighbors.get(chosen,()):
        command=key(proposals[expert])
        if command!=current:
            groups.setdefault(command,[]).append(expert)
    return groups
