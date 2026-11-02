"""Five refinements of the same single-robot north-push scene; no scene expansion."""
from dataclasses import replace
from experiments.option_testbench import single_push_speed_v2_eval as R
R.CONFIGS.clear()
for speed in (.3,.4,.5):
 R.CONFIGS[f'prepare_matched_{speed:.2f}']=replace(R.B,speed=.32,tangential_gain=4.8,tangential_speed=.16*.32/.24,point_rate=.16,contact_tangent_damping=.7,approach_tangential_speed=speed,approach_normal_speed=min(.3,speed),approach_tangential_gain=4.5,approach_velocity_damping=1.2,approach_brake_accel=.7)
R.CONFIGS['prepare_radial']=replace(R.CONFIGS['prepare_matched_0.40'],gap_gain=4.)
R.CONFIGS['prepare_damping']=replace(R.CONFIGS['prepare_matched_0.40'],approach_tangential_gain=5.5,approach_velocity_damping=1.5)
R.F.VARIANTS.update(R.CONFIGS)
if __name__=='__main__':R.main()
