"""Reference only: goal distance divided by nominal speed. No learned heuristic."""
import math


def score(context, nodes, memory):
    result=[]
    h=context.get('hypothesis')
    goal=context['observation']['robots'][0]['goal_xy']
    for node in nodes:
        prediction=node.get('prediction') or node.get('parent_prediction')
        if prediction:
            obj=prediction['state']['object'];x,y=obj['x'],obj['y']
        elif h:x,y=h['object'][:2]
        else:result.append(0.);continue
        result.append(math.hypot(x-goal[0],y-goal[1])/.3)
    return result
