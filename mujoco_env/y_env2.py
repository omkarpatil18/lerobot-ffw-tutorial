import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw


class SimpleEnv2:
    def __init__(
        self,
        xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
        seed=None,
    ):
        self.env = MuJoCoParserClass(name="Tabletop", rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names_l = [f"arm_l_joint{i}" for i in range(1, 8)]
        self.joint_names_r = [f"arm_r_joint{i}" for i in range(1, 8)]
        self.joint_names = self.joint_names_l + self.joint_names_r
        self.ctrl_names = self.joint_names + ["gripper_l_joint1", "gripper_r_joint1"]
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        self.env.reset()
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay="top right",
        )

    def reset(self, seed=None):
        if seed != None:
            np.random.seed(seed=0)
        # Home configuration (FFW bimanual)
        q_zero = np.array([-0.5, 0, 0, -0.5, 0, 0, 0] * 2)
        idxs = self.env.get_idxs_fwd(joint_names=self.joint_names)
        self.env.data.qpos[idxs] = q_zero
        self.env.data.qvel[:] = 0.0

        # Set object positions
        obj_names = self.env.get_body_names(prefix="body_obj_")
        n_obj = len(obj_names)
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range=[-0.35, -0.40],
            y_range=[-0.15, +0.15],
            z_range=[0.82, 0.82],
            min_dist=0.2,
            xy_margin=0.0,
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(
                body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :]
            )
            self.env.set_R_base_body(body_name=obj_names[obj_idx], R=np.eye(3, 3))
        self.env.forward(increase_tick=False)

        self.last_q = copy.deepcopy(q_zero)
        self.q = np.zeros(len(self.ctrl_names))
        self.q[: len(self.joint_names)] = q_zero
        self.p0_l, self.R0_l = self.env.get_pR_body(body_name="tcp_link_l")
        self.p0_r, self.R0_r = self.env.get_pR_body(body_name="tcp_link_r")
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate(
            [mug_red_init_pose, mug_blue_init_pose, plate_init_pose], dtype=np.float32
        )
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_l_state = False
        self.gripper_r_state = False
        self.past_chars = []

    def set_instruction(self, given=None):
        if given is None:
            self.instruction = "Pick up both the red and blue mugs and place them on the plate."
        else:
            self.instruction = given

    def step(self, action):
        if self.action_type == "eef_pose":
            # action: [dx_l,dy_l,dz_l,dr_l,dp_l,dyw_l,gripper_l, dx_r,dy_r,dz_r,dr_r,dp_r,dyw_r,gripper_r]
            self.p0_l += action[:3]
            self.R0_l = self.R0_l.dot(rpy2r(action[3:6]))
            q_l, _, _ = solve_ik(
                env=self.env,
                joint_names_for_ik=self.joint_names_l,
                body_name_trgt="tcp_link_l",
                q_init=self.env.get_qpos_joints(joint_names=self.joint_names_l),
                p_trgt=self.p0_l,
                R_trgt=self.R0_l,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
            self.p0_r += action[7:10]
            self.R0_r = self.R0_r.dot(rpy2r(action[10:13]))
            q_r, _, _ = solve_ik(
                env=self.env,
                joint_names_for_ik=self.joint_names_r,
                body_name_trgt="tcp_link_r",
                q_init=self.env.get_qpos_joints(joint_names=self.joint_names_r),
                p_trgt=self.p0_r,
                R_trgt=self.R0_r,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
            q = np.concatenate([q_l, q_r])
            gripper_l_cmd = action[6]
            gripper_r_cmd = action[13]
        elif self.action_type == "delta_joint_angle":
            q = action[: len(self.joint_names)] + self.last_q
            gripper_l_cmd = action[-2]
            gripper_r_cmd = action[-1]
        elif self.action_type == "joint_angle":
            q = action[: len(self.joint_names)]
            gripper_l_cmd = action[-2]
            gripper_r_cmd = action[-1]
        else:
            raise ValueError("action_type not recognized")

        self.compute_q = q
        self.q = np.concatenate([q, [gripper_l_cmd, gripper_r_cmd]])

        if self.state_type == "joint_angle":
            return self.get_joint_state()
        elif self.state_type == "ee_pose":
            return self.get_ee_pose()
        elif self.state_type == "delta_q" or self.action_type == "delta_joint_angle":
            return self.get_delta_q()
        else:
            raise ValueError("state_type not recognized")

    def step_env(self):
        self.env.step(ctrl=self.q, ctrl_names=self.ctrl_names)

    def grab_image(self):
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name="agentview")
        self.rgb_top = self.env.get_fixed_cam_rgb(cam_name="topview")
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name="sideview")
        return self.rgb_agent, self.rgb_top, self.rgb_side

    def render(self, teleop=False, idx=0):
        self.env.plot_time()
        for body_name, color in [
            ("tcp_link_l", [0.95, 0.05, 0.05, 0.5]),
            ("tcp_link_r", [0.05, 0.05, 0.95, 0.5]),
        ]:
            p_current, R_current = self.env.get_pR_body(body_name=body_name)
            self.env.plot_sphere(p=p_current, r=0.02, rgba=color)
            self.env.plot_capsule(
                p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05, 0.95, 0.05, 0.5]
            )
        rgb_top_view = add_title_to_img(self.rgb_top, text="Top View", shape=(640, 480))
        rgb_agent_view = add_title_to_img(self.rgb_agent, text="Agent View", shape=(640, 480))
        self.env.plot_T(
            p=np.array([0.1, 0.0, 1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False
        )
        self.env.viewer_rgb_overlay(rgb_agent_view, loc="top right")
        self.env.viewer_rgb_overlay(rgb_top_view, loc="bottom right")
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side, text="Side View", shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc="top left")
            self.env.viewer_text_overlay(
                text1="Key Pressed", text2="%s" % (self.env.get_key_pressed_list())
            )
            self.env.viewer_text_overlay(
                text1="Key Repeated", text2="%s" % (self.env.get_key_repeated_list())
            )
        if getattr(self, "instruction", None) is not None:
            self.env.viewer_text_overlay(text1="Language Instructions", text2=self.instruction)
        self.env.render()

    def get_joint_state(self):
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper_l = self.env.get_qpos_joint("gripper_l_joint1")
        gripper_r = self.env.get_qpos_joint("gripper_r_joint1")
        gripper_l_cmd = 1.0 if gripper_l[0] > 0.5 else 0.0
        gripper_r_cmd = 1.0 if gripper_r[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_l_cmd, gripper_r_cmd]], dtype=np.float32)

    def teleop_robot(self):
        """
        Bimanual teleoperation (FFW).

        LEFT ARM: W/S (x), A/D (y), R/F (z), Q/E (yaw), UP/DOWN (pitch), LEFT/RIGHT (roll), SPACE (gripper)
        RIGHT ARM: I/K (x), J/L (y), U/O (z), ,/. (yaw), HOME/END (pitch), PGUP/PGDN (roll), ENTER (gripper)
        Z: reset
        """
        step = 0.025
        rot = 0.1 * 0.5

        dpos_l = np.zeros(3)
        drot_l = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S): dpos_l += [step, 0, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W): dpos_l += [-step, 0, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A): dpos_l += [0, -step, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D): dpos_l += [0, step, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R): dpos_l += [0, 0, step]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F): dpos_l += [0, 0, -step]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q): drot_l = rotation_matrix(rot, [0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E): drot_l = rotation_matrix(-rot, [0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP): drot_l = rotation_matrix(-rot, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN): drot_l = rotation_matrix(rot, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT): drot_l = rotation_matrix(rot, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT): drot_l = rotation_matrix(-rot, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE): self.gripper_l_state = not self.gripper_l_state

        dpos_r = np.zeros(3)
        drot_r = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_K): dpos_r += [step, 0, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_I): dpos_r += [-step, 0, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_J): dpos_r += [0, -step, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_L): dpos_r += [0, step, 0]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_U): dpos_r += [0, 0, step]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_O): dpos_r += [0, 0, -step]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_COMMA): drot_r = rotation_matrix(rot, [0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_PERIOD): drot_r = rotation_matrix(-rot, [0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_HOME): drot_r = rotation_matrix(-rot, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_END): drot_r = rotation_matrix(rot, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_PAGE_UP): drot_r = rotation_matrix(rot, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_PAGE_DOWN): drot_r = rotation_matrix(-rot, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_ENTER): self.gripper_r_state = not self.gripper_r_state

        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(14, dtype=np.float32), True

        action = np.concatenate(
            [
                dpos_l,
                r2rpy(drot_l),
                [float(self.gripper_l_state)],
                dpos_r,
                r2rpy(drot_r),
                [float(self.gripper_r_state)],
            ],
            dtype=np.float32,
        )
        return action, False

    def get_delta_q(self):
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper_l = self.env.get_qpos_joint("gripper_l_joint1")
        gripper_r = self.env.get_qpos_joint("gripper_r_joint1")
        gripper_l_cmd = 1.0 if gripper_l[0] > 0.5 else 0.0
        gripper_r_cmd = 1.0 if gripper_r[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_l_cmd, gripper_r_cmd]], dtype=np.float32)

    def check_success(self):
        """
        Check if both the red and blue mugs are placed on the plate.
        Both grippers should be open and both TCPs above 0.9.
        """
        p_mug_red = self.env.get_p_body("body_obj_mug_5")
        p_mug_blue = self.env.get_p_body("body_obj_mug_6")
        p_plate = self.env.get_p_body("body_obj_plate_11")
        gripper_l_open = self.env.get_qpos_joint("gripper_l_joint1")[0] < 0.1
        gripper_r_open = self.env.get_qpos_joint("gripper_r_joint1")[0] < 0.1
        red_on_plate = (
            np.linalg.norm(p_mug_red[:2] - p_plate[:2]) < 0.1
            and np.linalg.norm(p_mug_red[2] - p_plate[2]) < 0.6
        )
        blue_on_plate = (
            np.linalg.norm(p_mug_blue[:2] - p_plate[:2]) < 0.1
            and np.linalg.norm(p_mug_blue[2] - p_plate[2]) < 0.6
        )
        if red_on_plate and blue_on_plate and gripper_l_open and gripper_r_open:
            p_l = self.env.get_p_body("tcp_link_l")[2]
            p_r = self.env.get_p_body("tcp_link_r")[2]
            if p_l > 0.9 and p_r > 0.9:
                return True
        return False

    def get_obj_pose(self):
        p_mug_red = self.env.get_p_body("body_obj_mug_5")
        p_mug_blue = self.env.get_p_body("body_obj_mug_6")
        p_plate = self.env.get_p_body("body_obj_plate_11")
        return p_mug_red, p_mug_blue, p_plate

    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        self.env.set_p_base_body(body_name="body_obj_mug_5", p=p_mug_red)
        self.env.set_R_base_body(body_name="body_obj_mug_5", R=np.eye(3, 3))
        self.env.set_p_base_body(body_name="body_obj_mug_6", p=p_mug_blue)
        self.env.set_R_base_body(body_name="body_obj_mug_6", R=np.eye(3, 3))
        self.env.set_p_base_body(body_name="body_obj_plate_11", p=p_plate)
        self.env.set_R_base_body(body_name="body_obj_plate_11", R=np.eye(3, 3))
        self.step_env()

    def get_ee_pose(self):
        p_l, R_l = self.env.get_pR_body(body_name="tcp_link_l")
        p_r, R_r = self.env.get_pR_body(body_name="tcp_link_r")
        return np.concatenate([p_l, r2rpy(R_l), p_r, r2rpy(R_r)], dtype=np.float32)
