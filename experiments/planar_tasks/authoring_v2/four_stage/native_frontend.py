"""Coarse native evidence with shared body envelopes and no union per candidate."""
import math

from shapely.geometry import MultiPoint

from .frontend import PublicFrontend
from .native_geometry import NativeEvidenceMap
from ..symbolic_unknown.geometry import disc, envelopes


class NativeFrontend(PublicFrontend):
    def __init__(self, *, resolution=.01, scan_stride_ticks=16, semantic_names=None):
        super().__init__(evidence=NativeEvidenceMap(resolution=resolution,
                                                   scan_stride_ticks=scan_stride_ticks),semantic_names=semantic_names)
        self._geometry_tick = None
        self._bodies = {}
        # Retain static free-space evidence, but do not repeatedly union tiny
        # floating-point variants of an object's inner polygon. The current
        # public CAD fit supplies a bounded-size outer pose set every boundary.
        # Keep the outer-set intersection (needed when the visible face is
        # incomplete), but keep only the larger previously certified inner set.
        self.refine_stationary_pose = "outer_only"

    def motion_regions(self, o, pose, opt):
        geometry_key=(o["tick"],self.target_stationary)
        if self._geometry_tick != geometry_key:
            self._bodies = {}
            self._geometry_tick = geometry_key
            self._brake = envelopes(o,pose,"brake",0,self.contract,
                                    target_stationary=self.target_stationary)
        robots = list(self._brake["robots"])
        target = self._brake["target"]
        kind = opt["family"]
        for i in opt["robots"]:
            key = (kind,i,opt["ticks"],opt["heading"] if kind=="move" else None)
            if key not in self._bodies:
                own=o["robots"][i]["own"]
                if kind=="push":
                    radius=self.contract.robot_radius+(self.contract.robot_stop_path+self.contract.push_robot_path*opt["ticks"]/4)
                    self._bodies[key]=disc(own["x"],own["y"],radius)
                elif kind=="orbit":
                    radius=self.contract.robot_radius+self.contract.robot_stop_path+.4*opt["ticks"]*.05
                    self._bodies[key]=disc(own["x"],own["y"],radius)
                elif kind=="move":
                    v=(own.get("vx",0.),own.get("vy",0.))
                    ref=o["robots"][i].get("execution",{}).get("reference_velocity") or (0.,0.)
                    u=(.2*math.cos(opt["heading"]*math.pi/4),.2*math.sin(opt["heading"]*math.pi/4))
                    vmax=max(math.hypot(*v),math.hypot(*ref),math.hypot(*u))
                    rmax=max(math.hypot(*ref),math.hypot(*u))
                    kmin=min(40.,15./max(vmax+rmax,1e-9))
                    horizon=opt["ticks"]*.05+rmax/.5+2*4./kmin+.1
                    points=[(own["x"]+horizon*q[0],own["y"]+horizon*q[1]) for q in ((0.,0.),v,ref,u)]
                    self._bodies[key]=MultiPoint(points).convex_hull.buffer((self.contract.robot_radius+.003)/math.cos(math.pi/128),quad_segs=32)
            robots[i]=self._bodies[key]
        if kind=="push" and pose is not None:
            key=("target",opt["ticks"])
            if key not in self._bodies:
                self._bodies[key]=pose.outer.buffer(self.contract.target_stop_boundary_path+
                                                  self.contract.target_push_boundary_path*opt["ticks"]/4)
            target=self._bodies[key]
        # Consumers of this profile query body pieces directly. No historical
        # or per-candidate union is needed on the deployment path.
        return dict(robots=robots,target=target,static=None)
