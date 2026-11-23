"""Frozen first-disagreement rules. Inputs contain only prefix-derived option descriptors."""
def choose(ga,attention):
    if {ga['kind'],attention['kind']} != {'push','move'}:
        return 'G','abstain'
    push,move=('G','T') if ga['kind']=='push' else ('T','G')
    drivers=(ga if push=='G' else attention).get('pushes')
    if not drivers:
        return 'G','abstain'
    for driver in drivers:
        if (driver['distance']<=.80 and driver['clearance']>=.40 and driver['alignment']>0):
            return push,'ready'
    return move,'reposition'


def action_only(ga,attention):
    if {ga['kind'],attention['kind']}=={'push','move'}:
        return 'G' if ga['kind']=='push' else 'T'
    return 'G'
