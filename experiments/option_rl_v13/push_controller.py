"""Continuous-mode PUSH: explicit versioned fork of the tested local controller step.

Only target completion is disabled. Safety, own-scan tracking, contact allocation,
preparation and feedback remain. This file is independent of frozen run11 sources.
"""
from __future__ import annotations
from dataclasses import replace
from coimagine.random_start import effect_push as P
from coimagine.random_start.skills import SkillObservation
from coimagine.random_start.effect_push import VERSION,_wrap_angle,add,asdict,availability,clamp,cross,dot,interpolate_face,math,mul,patch_gap,replace,spec,sub,tangent

#: Fitted-gap radius treated as contact-ready; loose enough for real localisation noise.
CONTACT_READY_GAP = 0.05
#: Extra clearance beyond contact-ready before declaring a lost contact.
CONTACT_RECOVERY_MARGIN = 0.02
#: Recovery only when the fitted gap exceeds this. With the current 5 cm
#: contact-ready tolerance this is 0.07 m, not 0.06 m.
RECOVERY_ENTER_GAP = max(0.06, CONTACT_READY_GAP + CONTACT_RECOVERY_MARGIN)


class ContinuousPush(P.EffectPushSkill):
    def __init__(self,action,force_limit,config):
        super().__init__(action,force_limit,config)
        self.time_budget=spec.TIME_BUDGET_SECONDS+spec.CONTROL_DT
        self.requested_time_budget=self.time_budget
        self.effect_last_observation_time=None
        self.effect_observation_dt=spec.CONTROL_DT
        original=self.reference.update
        def update(raw,dt):
            length=math.hypot(*raw)
            if length>.5:raw=tuple(v*.5/length for v in raw)
            return original(raw,dt)
        self.reference.update=update

    def _update_effect(self,face,dt):
        # A braked perception/peer pause does not generate intermediate evidence.
        # Estimate rates over the real observed interval when tracking resumes.
        return super()._update_effect(face,self.effect_observation_dt)

    def step(self, observation: SkillObservation, dt):
        ticks=(max(1,round((observation.time-self.effect_last_observation_time)/spec.CONTROL_DT))
               if self.effect_last_observation_time is not None else 1)
        self.effect_observation_dt=ticks*spec.CONTROL_DT
        self.effect_last_observation_time=observation.time
        self.skill_time += dt
        if self.stage != "recover":
            self.no_progress_seconds += dt
        self._note_position(observation)
        reason = self.pending_exit_reason or self._safety(observation)
        face = self._observe_face(observation)
        if face is not None and self.config.endpoint_tracking:
            tracked = self.endpoint_tracker.update(observation.scan, face.patch, observation.patches,
                                                   face.length, observation.time)
            if tracked is None:
                self.endpoint_status = "endpoint_unobservable"
                reason = reason or "effect_unobservable"
                face = replace(face, centre_observable=False)
            else:
                self.endpoint_status = tracked.status
                face = replace(face, point=tracked.midpoint,
                               centre=add(tracked.midpoint, mul(face.normal, face.depth / 2)),
                               centre_observable=tracked.status == "observed",
                               evidence="tracked_endpoint_plus_prior", tracking=tracked)
        normal_advance = self._measure_normal_travel(observation, face, dt)
        if self.normal_travel_mode and self.normal_travel > self.normal_progress_marker + .001:
            self.normal_progress_marker = self.normal_travel
            self._mark_progress()
        raw = (0.0, 0.0)
        if face is None:
            if self.face is None:
                _, unavailable = availability(observation, self.action, self.config)
                reason = reason or unavailable
            else:
                reason = reason or "effect_face_lost"
            self.effect_status = "unavailable" if self.face is None else "tracking_lost"
        else:
            self.face = face
            self.partial_geometry_time = (0.0 if face.centre_observable
                                          else self.partial_geometry_time + dt)
            if self.partial_geometry_time > self.config.partial_geometry_seconds:
                reason = reason or "effect_unobservable"
            local = replace(observation, patch=face.patch)
            gap = patch_gap(face.patch)
            if self.initial_face is None:
                self.initial_face = face
                # The effect belongs to the whole option, including preparation/contact
                # establishment. Do not reset its origin after the object already moved.
                self.reference_centre = face.centre
                self.initial_normal = self.last_normal = face.normal
                self.reference_time = observation.time
                self.heading_error = (math.atan2(cross(self.action.effect_direction, face.normal),
                                                dot(self.action.effect_direction, face.normal))
                                      if self.action.effect_kind == "translate" else 0.0)
                self.stage = "approach"
                self.effect_status = "approaching"
            self._update_effect(face, dt)
            self._choose_contact(face, dt, observation)
            reason = reason or self.contact_allocation_exit
            tangential_error = dot(sub(self.contact_target, (observation.own.x, observation.own.y)),
                                   tangent(face.normal))
            self.tangential_error = tangential_error
            if self.approach_blend is None:
                self.approach_blend = clamp((abs(tangential_error) - .05) / .10, 0., 1.)
            tangential_velocity = dot((observation.own.vx, observation.own.vy), tangent(face.normal))
            self.tangential_velocity = tangential_velocity
            self.first_clip = clamp(self.config.tangential_gain * tangential_error,
                                    -self.config.tangential_speed, self.config.tangential_speed)
            lateral = self.first_clip
            self.damping_term = 0.0
            if self.stage in ('push','follow'):
                self.damping_term = self.config.contact_tangent_damping * tangential_velocity
                lateral = clamp(lateral - self.damping_term,
                                -self.config.tangential_speed, self.config.tangential_speed)
            self.lateral_command = lateral
            if (self.recovery_enabled and self.joint_active and self.stage in ("push", "follow")
                    and gap > RECOVERY_ENTER_GAP):
                self.recovery_requested = True
            if self.config.fast_approach and self.approach_blend > 0 and self.stage in ("approach", "ready"):
                lateral = self._approach_lateral(tangential_error, tangential_velocity,
                                                 gap, observation.contacting_object)
            aligned = abs(_wrap_angle(math.atan2(face.normal[1], face.normal[0])
                                     - observation.own.angle)) <= math.radians(spec.COOP_READY_ALIGN_DEG)
            if self.stage in ("approach", "ready"):
                # Centre/offset on the observed face before committing normal push force.
                approach_gap = spec.COOP_READY_GAP if self.cooperative else self.config.solo_approach_gap
                radial = clamp(self.config.gap_gain * (gap - approach_gap), -0.08, self.config.approach_normal_speed)
                self.preparation_gap_target = approach_gap
                settled_laterally = (not self.config.fast_approach or self.approach_blend == 0 or
                                     abs(tangential_velocity) <= self.config.approach_settle_speed)
                if abs(tangential_error) > 0.025 or not settled_laterally:
                    if face.tracking is not None:
                        # Approach a safe staging line while moving towards the
                        # fixed endpoint-relative target; keep 3 cm extra clearance
                        # until tangential alignment, rather than waiting remotely.
                        extra=.03
                        if self.config.preparation_policy in ('blended','hold'):
                            extra *= clamp(max((abs(tangential_error)-.015)/.08,
                                               (abs(tangential_velocity)-self.config.approach_settle_speed)/.15),0.,1.)
                        self.preparation_gap_target=approach_gap+extra
                        radial = clamp(self.config.gap_gain * (gap - self.preparation_gap_target),
                                       -.08, self.config.approach_normal_speed)
                    else:
                        radial = min(0.0, radial)
                close = (gap <= spec.COOP_READY_GAP + 0.005 and abs(tangential_error) <= 0.025
                         and aligned and settled_laterally)
                if self.cooperative and close:
                    self.ready = True
                    self.stage = "ready"
                    radial, lateral = 0.0, 0.0
                elif self.cooperative and not self.release:
                    self.ready = False
                    self.stage = "approach"
                if (self.stage == "ready" and self.release) or (close and not self.cooperative):
                    self.stage = "push"
                    self.effect_status = "engaging"
                    if self.joint_pair:
                        self._mark_progress()  # readiness waiting is not failed contact seeking
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                if self.joint_pair:
                    preparation_error = abs(tangential_error) + max(0., gap - spec.COOP_READY_GAP)
                    if preparation_error < self.best_preparation_error - .005:
                        self.best_preparation_error = preparation_error
                        self._mark_progress()
                else:
                    self._mark_progress()
            if self.stage == "push":
                # At the surface the single-tick Pymunk contact flag can flicker
                # while the fitted gap is already ~0.  Treat a small positive gap
                # as contact-ready so the pair gate does not drop PUSH every other
                # tick; the force controller still closes the remaining gap.
                surface_contact = gap is not None and gap <= CONTACT_READY_GAP
                contact_now = bool(observation.contacting_object) or surface_contact
                self.contact_ready = contact_now
                if contact_now and not self.contact_seen:
                    self.effect_status = "executing"
                    self._mark_progress()
                if contact_now:
                    self.effective_contact += dt
                    self.contact_seen = True
                    self.contact_loss_seconds = 0.0
                    self._accumulate_signed_progress(self.action.effect_direction
                                                    if self.action.effect_kind == "translate" else face.normal)
                else:
                    self.contact_loss_seconds += dt
                    if (not self.recovery_enabled and self.contact_seen
                            and self.contact_loss_seconds > spec.CONTACT_LOSS_TOLERANCE):
                        reason = reason or "contact_lost"
                    elif not self.contact_seen and self.contact_loss_seconds > spec.CONTACT_SEEK_SECONDS:
                        reason = reason or "blocked_no_progress"
                remaining = 1.0  # No terminal effect target; speed below is the current mode command.
                speed = clamp(self.config.target_gain * remaining, self.config.min_push_speed,
                              self.speed_command)
                if self.rotation_trimming:
                    speed = clamp(self.config.target_gain * remaining,
                                  self.config.rotation_trim_min_speed, self.config.rotation_trim_speed)
                radial = clamp(speed + self.config.gap_gain * gap, 0.0, self.speed_command)
                if self.joint_pair and not self.joint_active:
                    # Establish contact at a common gentle, UNscaled speed. A side
                    # already touching holds, allowing its peer to catch up.
                    radial = (0.0 if observation.contacting_object else
                              clamp(self.config.joint_engage_speed + self.config.gap_gain * gap,
                                    0.0, self.config.joint_engage_speed))
                    self.effect_status = "joint_engaging"
                elif self.normal_reference_mode or self.normal_travel_mode:
                    # Actual own normal travel (or the explicitly selected legacy
                    # reference clock) determines completion, not object displacement.
                    radial = clamp(self.surface_normal_speed + self.speed_command + self.config.gap_gain * gap,
                                   0.0, self.config.speed)
                    if observation.contacting_object:
                        self.normal_reference_budget += self.speed_command * dt
                        if self.normal_reference_mode:
                            self._mark_progress()  # historical reference-budget contract
                    self.effect_status = "normal_travel_executing" if self.normal_travel_mode else "normal_reference_executing"
                if self.action.effect_kind == "rotate":
                    if not observation.contacting_object:
                        radial = min(radial, self.config.rotation_engage_speed)
                    elif self.cooperative:
                        radial = min(radial, self.config.cooperative_rotation_speed)
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                angular_coast = (max(0., self.action.rotation_sign * self.angular_rate,
                                     self.action.rotation_sign * self.instant_angular_rate)
                                 * self.config.rotation_brake_seconds
                                 if self.action.effect_kind == "rotate" else 0.)
                reached = False  # Continuous modes have no distance/angle completion.
                if self.reference_centre is not None and (not self.joint_pair or self.joint_active) and reached:
                    if reason is None and not self.drive_complete:
                        self.drive_complete = True
                        self.drive_complete_time = observation.time
                        self.drive_complete_travel = self.normal_travel
                    if self.follow_enabled and reason is None:
                        self.follow_started = True
                        self.stage = "follow"
                    else:
                        self.stage = "brake"
                        self.effect_status = "settling"
                if not self.normal_travel_mode and abs(self.effect_speed) > 0.005:
                    self._mark_progress()
            if self.stage == "follow":
                self.follow_seconds += dt
                if gap <= self.config.recovery_enter_gap:
                    self.follow_near_seconds += dt
                if observation.contacting_object:
                    self.follow_contact_seconds += dt
                    self.follow_contact_loss_seconds = 0.0
                else:
                    self.follow_contact_loss_seconds += dt
                if not self.recovery_enabled and self.follow_contact_loss_seconds > self.config.follow_contact_loss:
                    reason = reason or "follow_contact_lost"
                if self.follow_seconds >= self.config.follow_max_seconds:
                    reason = reason or "follow_time_limit"
                if self.follow_path_length >= self.config.follow_max_travel:
                    reason = reason or "follow_travel_limit"
                # Follow locally observed surface motion with a small gap correction.
                # There is no active propulsion bias or nominal-direction correction.
                radial = clamp(self.surface_normal_speed + clamp(
                    self.config.gap_gain * (gap - self.config.follow_gap),
                    -self.config.follow_gap_speed, self.config.follow_gap_speed),
                    -self.config.follow_speed, self.config.follow_speed)
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                self.effect_status = "contact_following"
                if self.follow_stop:
                    self.stage = "brake"
                    self.effect_status = "joint_settling"
            if self.stage == "recover":
                self.recovery_seconds += dt
                self.recovery_ready = (gap <= self.config.recovery_resume_gap
                    and abs(dot((observation.own.vx, observation.own.vy), face.normal)) <= .06)
                radial = (0.0 if gap <= self.config.recovery_resume_gap else
                          clamp(self.config.gap_gain * (gap - self.config.follow_gap),
                                0.0, self.config.recovery_approach_speed))
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                self.effect_status = "joint_reapproaching"
            if self.stage == "brake":
                raw = (0.0, 0.0)
                self.local_brake_seconds += dt
                settle_limit = (self.config.settle_speed if self.action.effect_kind == "translate"
                                else self.config.settle_angular_speed)
                settled_effect = self.joint_pair or self.normal_travel_mode or self.normal_reference_mode or abs(self.effect_speed) <= settle_limit
                if observation.own.speed <= self.config.settle_speed and settled_effect:
                    self.settled_seconds += dt
                else:
                    self.settled_seconds = 0.0
                if (self.settled_seconds >= self.config.settle_seconds
                        and (self.normal_travel_mode or self.normal_reference_mode or face.centre_observable
                             or self.action.effect_kind == "rotate")):
                    if self.normal_travel_mode:
                        if self.normal_travel >= self.action.distance - 1e-9:
                            reason = reason or "normal_travel_complete"
                            self.effect_status = "normal_travel_completed"
                        else:
                            # Contact recoil during braking must not turn a past
                            # threshold crossing into a false completion claim.
                            self.stage = "push"
                            self.settled_seconds = self.local_brake_seconds = 0.0
                            self.effect_status = "normal_travel_trimming"
                    elif self.normal_reference_mode:
                        reason = reason or "normal_reference_complete"
                        self.effect_status = "reference_completed"
                    elif self.effect_progress >= self.action.effect_target:
                        reason = reason or ("local_effect_reached" if self.joint_pair else "effect_complete")
                        self.effect_status = "completed"
                    else:
                        # Anticipatory braking is a control decision, not evidence that the
                        # requested effect happened. Resume a bounded slow trim if short.
                        self.stage = "push"
                        self.rotation_trimming = True
                        self.settled_seconds = 0.0
                        self.effect_status = "trimming"
                if (self.joint_pair or self.normal_travel_mode or self.normal_reference_mode) and self.local_brake_seconds >= spec.BRAKE_MAX_SECONDS:
                    reason = reason or "local_brake_timeout"
            if self.config.progress_guard and reason is None:
                phase=None;value=0.;minimum=.005
                if self.stage=='approach':
                    phase='prepare';value=-(abs(tangential_error)+max(0.,gap-spec.COOP_READY_GAP))
                elif self.stage=='push' and self.joint_pair and not self.joint_active:
                    if not observation.contacting_object:phase='engage';value=-max(0.,gap)
                elif self.stage=='push':
                    phase='translate' if self.action.effect_kind=='translate' else 'rotate'
                    value=self.normal_travel if self.normal_travel_mode else self.effect_progress
                    minimum=.005 if phase=='translate' else .01
                if self.phase_progress.update(phase,value,dt,minimum):
                    reason={'prepare':'prepare_stalled','engage':'engagement_stalled',
                            'translate':'translation_stalled','rotate':'rotation_stalled'}[phase]
            reason = reason or self._common_checks(local, dt)
            if (reason is None and self.stage == "push" and self.reference_centre is not None
                    and self.no_progress_seconds > spec.NO_PROGRESS_SECONDS):
                reason = "blocked_no_progress"
        self.radial_command = radial
        self.raw_before_avoid = raw
        if reason is not None:
            raw = (0.0, 0.0)
        else:
            raw = self._obstacle_avoidance(observation, raw, include_teammate=True)
        self.raw_after_avoid = raw
        self.raw_before_norm_cap = raw
        if self.stage == "follow" and reason is None:
            length = math.hypot(*raw)
            if length > self.config.follow_speed:
                raw = mul(raw, self.config.follow_speed / length)
        reference = self.reference.update(raw, dt)
        self.raw_after_norm_cap = raw
        self.v_ref = reference
        force, saturated = self._force_from_reference(reference, observation)
        self.force_pre_norm_cap = (spec.KV * (reference[0] - observation.own.vx),
                                   spec.KV * (reference[1] - observation.own.vy))
        self.force_post_norm_cap = force
        if self.stage == "follow" and reason is None:
            # This caps commanded force, not unmeasured physical contact impulse.
            normal_force = dot(force, face.normal)
            force = add(force, mul(face.normal, clamp(normal_force,
                -self.config.follow_normal_force, self.config.follow_normal_force) - normal_force))
            length = math.hypot(*force)
            if length > self.config.follow_force:
                force = mul(force, self.config.follow_force / length)
            saturated = saturated or abs(normal_force) > self.config.follow_normal_force or length > self.config.follow_force
        if reason is not None:
            self._finish(reason, observation)
        local = replace(observation, patch=face.patch) if face else observation
        info = super()._info(local, force, saturated, reference, raw)
        self.progress = self.effect_progress
        info.update(progress=self.effect_progress, progress_target=self.action.effect_target)
        if self.normal_travel_mode:
            info["completion_progress"] = self.normal_travel
            info["completion_target"] = self.action.distance
        info["effect"] = {
            "version": VERSION, "kind": self.action.effect_kind,
            "direction": list(self.action.effect_direction), "rotation_sign": self.action.rotation_sign,
            "target": self.action.effect_target, "unit": "m" if self.action.effect_kind == "translate" else "rad",
            "progress": self.effect_progress, "robot_travel": self.signed_progress,
            "heading_error": self.heading_error, "lateral_error": self.lateral_error,
            "angular_rate": self.angular_rate, "offset": self.point_offset,
            "role_offset": self.role_offset,
            "q_rot": self.rotation_feedback, "diagnostic_tangent_offset": self.diagnostic_tangent_offset,
            "target_before_visibility": self.target_before_visibility,
            "target_before_reservation": self.target_before_reservation,
            "target_final": self.target_final,
            "e_t": self.tangential_error, "v_t": self.tangential_velocity,
            "first_clip": self.first_clip, "c_v_t": self.damping_term,
            "lateral": self.lateral_command, "radial": self.radial_command,
            "raw_before_avoid": self.raw_before_avoid, "raw_after_avoid": self.raw_after_avoid,
            "raw_before_norm_cap": self.raw_before_norm_cap, "raw_after_norm_cap": self.raw_after_norm_cap,
            "v_ref": self.v_ref, "F_pre_norm_cap": self.force_pre_norm_cap,
            "F_post_norm_cap": self.force_post_norm_cap,
            "contact_target": self.contact_target, "estimated_centre": face.centre if face else None,
            "face_midpoint": face.point if face else None, "face_normal": face.normal if face else None,
            "face_points": face.patch.points_world if face else [],
            "status": self.effect_status, "feedback": self.config.feedback,
            "estimate": face.evidence if face else "unavailable", "auto_reposition": False,
            "centre_observable": bool(face and face.centre_observable),
            "geometry_age": self.partial_geometry_time,
            "observation_time": observation.time, "reference_time": self.reference_time,
            "reference_normal": self.initial_normal,
            "face_confidence": face.confidence if face else 0.0,
            "endpoint_tracking": self.config.endpoint_tracking,
            "endpoint_status": self.endpoint_status,
            "endpoint": face.tracking.as_dict() if face and face.tracking else None,
            "anchor_distance": self.anchor_distance,
            "fast_approach": self.config.fast_approach,
            "preparation_policy": self.config.preparation_policy,
            "preparation_gap_target": self.preparation_gap_target,
            "progress_guard": self.phase_progress.info if self.config.progress_guard else None,
            "approach_speed_limit": self.approach_speed_limit,
            "approach_blend": self.approach_blend,
            "joint_execution": self.execution_protocol if self.joint_pair else "single_option",
            "joint_active": self.joint_active,
            "normal_reference_budget": self.normal_reference_budget,
            "normal_reference_target": self.action.distance if self.normal_reference_mode or self.normal_travel_mode else None,
            "normal_surface_follow_speed": self.surface_normal_speed,
            "normal_travel": self.normal_travel,
            "normal_travel_target": self.action.distance if self.normal_travel_mode else None,
            "normal_travel_increment": normal_advance,
            "post_push": self.config.post_push,
            "drive_complete": self.drive_complete,
            "drive_complete_time": self.drive_complete_time,
            "drive_complete_travel": self.drive_complete_travel,
            "follow_enabled": self.follow_enabled,
            "follow_started": self.follow_started,
            "follow_stop": self.follow_stop,
            "follow_seconds": self.follow_seconds,
            "follow_contact_seconds": self.follow_contact_seconds,
            "follow_near_seconds": self.follow_near_seconds,
            "follow_normal_travel": self.follow_normal_travel,
            "follow_normal_increment": self.follow_increment,
            "follow_path_length": self.follow_path_length,
            "contact_recovery": self.config.contact_recovery,
            "recovery_enabled": self.recovery_enabled,
            "recovery_requested": self.recovery_requested,
            "recovery_ready": self.recovery_ready,
            "recovery_resume_stage": self.recovery_resume_stage,
            "recovery_count": self.recovery_count,
            "recovery_seconds": self.recovery_seconds,
            "recovery_path_length": self.recovery_path_length,
            "recovery_normal_travel": self.recovery_normal_travel,
            "recovery_normal_increment": self.recovery_increment,
            "scale_semantics": ("measured_normal_travel_m" if self.normal_travel_mode else
                                "normal_reference_m" if self.normal_reference_mode else "estimated_object_effect"),
            "same_face": asdict(self.reservation_result) if self.reservation_result else None,
            "contact_allocation_mode": ("mixed-peer-occupied-v1" if self.config.mixed_contact_allocation
                                        and not self.cooperative else "cooperative" if self.cooperative else "off"),
            "tangential_error": self.tangential_error,
            "interpolated_face_points": (interpolate_face(face.patch, self.config.interpolation_factor)
                                         if face and self.config.endpoint_tracking else []),
        }
        self.travel_previous_normal = face.normal if face is not None else None
        self.travel_previous_contact = bool(observation.contacting_object)
        self.travel_previous_active = (self.stage in ("push", "brake", "follow") and not self.done
                                       and (not self.joint_pair or self.joint_active))
        self.travel_previous_follow = self.follow_started and self.stage in ("follow", "brake") and not self.done
        self.travel_previous_recovery = self.stage == "recover" and not self.done
        self.travel_previous_time = observation.time
        return force, info
