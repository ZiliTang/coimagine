from experiments.option_rl_v12.handoff import eligible

def test_handoff_requires_actual_clean_exit_and_does_not_restart_faults():
 assert not eligible(None,None)
 assert not eligible({'status':'wall_budget_reached'},None)
 assert eligible({'status':'wall_budget_reached'},{'exit_code':0})
 assert not eligible({'status':'wall_budget_reached'},{'exit_code':1})
 assert not eligible({'status':'requested_stop','stop_reason':'nonfinite_checkpoint'},{'exit_code':0})
 assert eligible({'status':'requested_stop','stop_reason':'agent_bounded_plateau_review_6000_no_gain_in_two_windows'},{'exit_code':0})
