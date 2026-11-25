def module_update(local,event,memory):
    o=event['observation']
    tick=o['tick']
    dt=event.get('executed_ticks') or 0
    action=event.get('action')
    c=memory.get('compact')
    tid=o['goal']['target_id']
    plan=local.get('plan')
    last=local.pop('proposal',None)
    pending=local.pop('pending',None)
    if plan is None and pending and dt and action==last:
        plan=local['plan']=pending

    # Retain the static surfaces seen during approaches, including those now occluded.
    cells=local.get('cells',[])
    fixed=obstacle_points(o,tid,True)
    observed={round(p[0]*10)*64+round(p[1]*10) for p in fixed}
    if observed:
        cells=[k for k in cells if k not in observed]+sorted(observed)
        local['cells']=cells[-150:]

    valid=bool(c and get_points(o,tid))
    samples=[s for s in local.get('samples',[]) if tick-s[0]<290]
    if valid and dt:
        stalled=0
        if action and action['target_id']==tid:
            for i,r in enumerate(o['robots']):
                if action['a%d'%i]>=32:
                    continue
                stage=r['execution'].get('stage')
                if stage in ('push','wait_peer_contact'):
                    own=r['own']
                    if r['contacting_static'] or (r['contacting_object'] and own['vx']**2+own['vy']**2<.007):
                        stalled=dt
        samples.append([tick,c[0],c[1],stalled])
    local['samples']=samples[-40:]

    if plan:
        if not valid or tick-plan['start']>235:
            local.pop('plan')
        else:
            d=DIR[plan['direction']]
            advance=dot(sub(c,plan['center']),d)
            i=plan['robot']
            if advance>plan['best']+.070:
                plan['best']=advance
                plan['progress_tick']=tick
            if action and action==last and action['target_id']==tid:
                r=o['robots'][i]
                braced_progress=(plan.get('through') and tick-plan['progress_tick']<=21)
                if r['contacting_static'] and not braced_progress:
                    plan['blocked']=plan.get('blocked',0)+dt
                else:
                    plan['blocked']=0
            minimum_reached=advance>=plan['length'] and not plan.get('through')
            if (minimum_reached or advance>=1.32 or advance<-.21 or abs(cross(sub(c,plan['center']),d))>.39
                or tick-plan['progress_tick']> (91 if plan['best']>.12 else 135)
                or plan.get('blocked',0)>30):
                local.pop('plan')
        if 'plan' not in local:
            local['release']=[tick,*plan['center']]
            return local

    packet=module_act(local,memory,o,None)
    if packet:
        if plan:
            local['proposal']=packet
        else:
            local['proposal'],local['pending']=packet
    elif plan:
        local.pop('plan')
        local['release']=[tick,*plan['center']]
    return local


def module_act(local,memory,observation,base_action):
    o=observation
    c=memory.get('compact')
    tick=o['tick']
    tid=o['goal']['target_id']
    goal=o['goal']['xy']
    if not c or o['active_target']!=tid or not get_points(o,tid):
        return None
    plan=local.get('plan')
    if base_action is not None:
        proposal=local.get('proposal')
        if plan or not proposal:
            return proposal
        if local.get('_force',False):
            return proposal
        if any(r['contacting_object'] for r in o['robots']):
            return None
        hist=local.get('samples',[])
        old=[s for s in hist if tick-s[0]>=165]
        if (not old or mag(sub(c,old[-1][1:3]))>.17 or
                memory['stale_ticks']<165 or memory['progress']>.027):
            return None
        p=local['pending']
        d=DIR[p['direction']]
        direct=norm(sub(goal,c))
        detour=dot(d,direct)<.68
        # Repeated physical failure is needed before substituting a forward drive
        # similar to one already available to the frozen policy.
        if sum(s[3] for s in hist)<(65 if detour else 100):
            return None
        if local.get('release') and (tick-local['release'][0]<205 or
                                    mag(sub(c,local['release'][1:]))<.32):
            return None
        return proposal

    robots=o['robots']
    poses=[position(r) for r in robots]
    arena=robots[0]['arena']
    pairs=o['entity_legal_pairs'].get(tid,[])
    axes=(c[2:4],(-c[3],c[2]))
    direct=norm(sub(goal,c))
    static=obstacle_points(o,tid,True)
    static+= [(k//64*.10,k%64*.10) for k in local.get('cells',[])]
    obstacles=obstacle_points(o,tid)+static
    def edge(p):
        return min(p[0],p[1],arena[0]-p[0],arena[1]-p[1])
    def box_gap(p):
        q=sub(p,c)
        return mag(tuple(max(0.,abs(dot(u,q))-.5*w) for u,w in zip(axes,c[4:])))
    def segment_free(p,q,extra,radius):
        if edge(q)<.22:
            return False
        for x in extra:
            if segment_near(x,p,q)<min(radius,mag(sub(x,p))-.018):
                return False
        return all(box_gap(shift(p,sub(q,p),f))>=min(radius,box_gap(p)-.017)
                   for f in (.28,.56,.8,1.))
    def steer(i,dst,other):
        p=poses[i]
        returns=hits(robots[i])
        selected=None
        score=.04
        for j,u in enumerate(DIR):
            gap=clearance(p,u,arena,returns)
            advance=dot(sub(dst,p),u)
            if gap<.055 or advance<.055:
                continue
            q=shift(p,u,min(.32,gap,advance))
            if not segment_free(p,q,other,.20):
                continue
            rank=mag(sub(p,dst))-mag(sub(q,dst))
            if rank>score:
                speed=3 if advance>.41 and gap>.38 else 2 if advance>.24 and gap>.23 else 1
                selected=47+4*j+speed
                score=rank
        if selected is None:
            return None
        return make(None,selected if i==0 else 44,44 if i==0 else selected,29)
    def station(i,k):
        d=DIR[k]
        r=robots[i]
        p=poses[i]
        peer=poses[1-i]
        f=face_for(visible_faces(r),d)
        if f is None and not any(abs(dot(d,u))>.94 for u in axes):
            return None
        n=d if f is None else f[0]
        half=.5*sum(abs(dot(n,u))*w for u,w in zip(axes,c[4:]))
        q=shift(c,n,-half) if f is None else face_q(f)
        stage=shift(q,n,-.31)
        offset=sub(p,q)
        if mag(sub(p,stage))>1.14 or edge(stage)<.23:
            return None
        push=[44,44]
        push[i]=4*k+3
        peer_gap=segment_near(peer,stage,shift(q,n,.40))
        if (f and push in pairs and -.77<dot(offset,n)<-.12 and abs(cross(offset,n))<.39
            and not r['contacting_static']):
            if peer_gap>.39:
                return make(tid,push[0],push[1],33)
            push[1-i]=46
            if push in pairs and peer_gap>.23:
                return make(tid,push[0],push[1],33)
        if peer_gap<.39:
            back=norm(sub(peer,q))
            tangent=(-n[1],n[0])
            options=[]
            for side in (tangent,(-tangent[0],-tangent[1]),back):
                dst=shift(peer,side,.38)
                free=segment_near(dst,stage,shift(q,n,.40))
                if free>.42 and segment_free(peer,dst,[p]+obstacles,.22):
                    options.append((free,dst))
            if options:
                cmd=steer(1-i,max(options)[1],[p]+obstacles)
                if cmd:
                    return cmd
            return None
        if any(mag(sub(stage,x))<.20 for x in obstacles) or box_gap(stage)<.16:
            return None
        if r['contacting_static'] and mag(sub(p,stage))<.20:
            stage=shift(stage,sub(stage,q),.50)
        points=[stage]
        tangent=(-n[1],n[0])
        span=.5*sum(abs(dot(tangent,u))*w for u,w in zip(axes,c[4:]))
        for sign in (-1,1):
            corner=shift(stage,tangent,sign*(span+.24))
            if segment_free(corner,stage,[peer]+obstacles,.24):
                points.append(corner)
        for dst in points:
            if mag(sub(p,dst))<.09 or not segment_free(p,dst,[peer]+obstacles,.215):
                continue
            wanted=sub(dst,p)
            if (dst==stage and mag(wanted)>.46 and abs(dot(wanted,n))<.26*mag(wanted)):
                m=move(r,dst)
                if m>=47:
                    u=DIR[(m-47)//4]
                    qmove=shift(p,u,min(.34,clearance(p,u,arena,hits(r))))
                    if dot(u,norm(wanted))>.60 and segment_free(p,qmove,[peer]+obstacles,.215):
                        return make(None,m if i==0 else 44,44 if i==0 else m,29)
            cmd=steer(i,dst,[peer]+obstacles)
            if cmd:
                return cmd
        return None

    if plan:
        i,k=plan['robot'],plan['direction']
        d=DIR[k]
        free=safe_travel(c,d,obstacles,arena)
        if free<.11:
            return None
        if plan.get('through') and plan['best']>=plan['length']-.07:
            if dot(d,direct)<.56:
                alternatives=[u for u in DIR if dot(u,direct)>.53 and abs(dot(d,u))<.34
                              and any(abs(dot(u,a))>.92 for a in axes)]
                if any(safe_travel(c,u,obstacles,arena)>.72 for u in alternatives):
                    return None
            r=robots[i]
            peer=poses[1-i]
            p=poses[i]
            push=[44,44]
            push[i]=4*k+3
            if (push in pairs and free>.20 and tick-plan['progress_tick']<=39
                    and r['execution'].get('stage')=='push'
                    and segment_near(peer,p,shift(p,d,.65))>.39):
                return make(tid,push[0],push[1],26)
        return station(i,k)
    if robots[0]['remaining_seconds']<12.0 or mag(sub(goal,c))<.72:
        return None
    direct_gap=safe_travel(c,direct,static,arena)
    if direct_gap>.72:
        return None
    if not any(dot(f[0],direct)>.49 and -.75<dot(sub(p,face_q(f)),f[0])<-.10
               and abs(cross(sub(p,face_q(f)),f[0]))<.40
               for r,p in zip(robots,poses) for f in visible_faces(r)):
        return None
    best=None
    for k,d in enumerate(DIR):
        alignment=dot(d,direct)
        if alignment<-.10 or not any(abs(dot(d,u))>.93 for u in axes):
            continue
        safe=safe_travel(c,d,obstacles,arena)
        for length in (.36,.50,.66):
            if safe<max(length+.115,direct_gap+.20):
                continue
            nxt=[*shift(c,d,length),*c[2:]]
            onward=safe_travel(nxt,norm(sub(goal,nxt)),obstacles,arena)
            if onward<max(.61,direct_gap+.24):
                continue
            for i in range(2):
                cmd=station(i,k)
                if not cmd:
                    continue
                score=.93*min(1.1,onward)+.53*alignment-.30*length
                half=.5*sum(abs(dot(d,u))*w for u,w in zip(axes,c[4:]))
                rear=shift(c,d,-half-.31)
                score-=.27*mag(sub(poses[i],rear))
                score-=.44*max(0.,dot(sub(poses[i],rear),d)-.27)
                if cmd['target_id']==tid:
                    score+=.36
                elif cmd['a%d'%i]==44:
                    score-=.16
                if best is None or score>best[0]:
                    best=(score,cmd,{'center':list(c[:2]),'robot':i,'direction':k,'length':length,
                                     'start':tick,'progress_tick':tick,'best':0.,
                                     'through':alignment>.69})
    return [best[1],best[2]] if best else None
