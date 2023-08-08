"A simplified manipulation environment for objects."

import os

import gym
from gym import spaces
from gym.envs.mujoco import MujocoEnv

import random
import numpy as np

initial_states = np.array([[0.0, 0.0, 2.5, 0.0, -1., -1.]])
goal_states = np.array([[0.0, 0.0, -2.5, -1.0, -1., -1.],
                        [0.0, 0.0, -2.5,  1.0, -1., -1.],
                        [0.0, 0.0,  0.0,  2.0, -1., -1.],
                        [0.0, 0.0,  0.0, -2.0, -1., -1.],
                       ])


class TabletopManipulation(MujocoEnv):

  FILE_TREE = "tabletop_manipulation.xml"
  MODEL_PATH_TREE = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), "tabletop_assets", FILE_TREE)

  def __init__(self,
               task_list="rc_r-rc_k-rc_g-rc_b",
               reward_type="dense",
               reset_at_goal=False,
               wide_init_distr=True):
    self.object_colors = [
        "r",
    ]
    self.objects = [
        "c",
    ]
    self.target_colors = ["r", "g", "b", "k"]

    # Dict of object to index in qpos
    self.object_dict = {
        (0, 0): [2, 3],
    }

    self.attached_object = (-1, -1)
    self.threshold = 0.4
    self.move_distance = 0.2

    self._task_list = task_list
    self._reward_type = reward_type
    self.initial_state = initial_states.copy()[0]
    self._goal_list = goal_states.copy()
    self.goal = self.initial_state.copy()
    self._reset_at_goal = reset_at_goal  # use only in train envs without resets
    self._wide_init_distr = wide_init_distr
    super().__init__(model_path=self.MODEL_PATH_TREE, frame_skip=15)

  def _get_obs(self):
    return np.concatenate([
        self.sim.data.qpos.flat[:4],  # remove the random joint
        np.asarray(self.attached_object),
        self.goal,
    ]).astype("float32")

  def get_next_goal(self):
    # the gripper should return to the original position (def in sparse reward)
    goal = self.initial_state.copy()

    cur_task = random.sample(self._task_list.split("-"), 1)[0]
    for task in cur_task.split("__"):
      color_to_move = self.object_colors.index(task.split("_")[0][0])
      object_to_move = self.objects.index(task.split("_")[0][1])
      target_index = self.target_colors.index(task.split("_")[1])

      obj_idx = self.object_dict[(color_to_move, object_to_move)]
      target_pos = self._goal_list[target_index][2:4]
      goal[obj_idx[0]:obj_idx[1] + 1] = target_pos  # the object

    return goal

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()
    self.goal = goal

    # # visualize goal
    # target_fist_pos = self.goal[0:2]
    # target_redcube_pos = self.goal[2:4]
    # self.sim.model.site_pos[self.model.site_name2id("goal:fist")][:2] = target_fist_pos
    # self.sim.model.site_pos[self.model.site_name2id("goal:redcube")][:2] = target_redcube_pos

  def set_state(self, qpos, qvel=None):
    del qvel
    qpos = np.concatenate([qpos[:4], np.array([-10])])
    qvel = self.sim.data.qvel.copy()
    super().set_state(qpos, qvel)

    if self.attached_object != (-1, -1): # if fist has grasped an object
      self.sim.model.geom_rgba[self.sim.model.geom_name2id('pointbodyr:attached')][3] = 1.0
      self.sim.model.geom_rgba[self.sim.model.geom_name2id('pointbodyr:detached')][3] = 0.0
    else: # if fist has not grasped an object
      self.sim.model.geom_rgba[self.sim.model.geom_name2id('pointbodyr:attached')][3] = 0.0
      self.sim.model.geom_rgba[self.sim.model.geom_name2id('pointbodyr:detached')][3] = 1.0

  def is_valid_init(self, state, goals):
    if np.linalg.norm(state[0:2]-state[2:4]) < 1:
      return False

    for g in goals:
      if np.linalg.norm(state[2:4] - g[2:4]) < 1:
        return False

    return True

  def is_valid_init_rev(self, state, goals):
    if np.linalg.norm(state[0:2]-state[2:4]) < 1:
      return False

    return True

  def reset(self):
    self.attached_object = (-1, -1)
    full_qpos = np.zeros((4,))

    if self._reset_at_goal:
      self.reset_goal()
      full_qpos[:4] = self.goal[:4]
      # full_qpos[:4] = np.random.uniform(-2.5, 2.5, size=(4,))
    else:
      if self._wide_init_distr:
        full_qpos[:4] = np.random.uniform(-2.5, 2.5, size=(4,))
        while not self.is_valid_init(full_qpos[:4], goal_states):
          full_qpos[:4] = np.random.uniform(-2.5, 2.5, size=(4,))
      else:
        full_qpos[:4] = self.initial_state[:4]

      self.reset_goal()

    self.set_state(full_qpos)
    self.sim.forward()
    return self._get_obs()

  def step(self, action):
    # rescale and clip action
    action = np.clip(action, np.array([-1.] * 3), np.array([1.] * 3))
    lb, ub = np.array([-0.2, -0.2, -0.2]), np.array([0.2, 0.2, 0.2])
    action = lb + (action + 1.) * 0.5 * (ub - lb)

    self.move(action)
    next_obs = self._get_obs()
    reward = self.compute_reward(next_obs)
    done = False
    info = {'is_success': self.is_successful(obs=next_obs)}
    return next_obs, reward, done, info

  def move(self, action):
    current_fist_pos = self.sim.data.qpos[0:2].flatten()
    curr_action = action[:2]

    if action[-1] > 0:
      if self.attached_object == (-1, -1):
        self._dist_of_cur_held_obj = np.inf  # to ensure the closest object is grasped when multiple objects are within threshold
        for k, v in self.object_dict.items():
          curr_obj_pos = np.array([self.sim.data.qpos[i] for i in v])
          dist = np.linalg.norm((current_fist_pos - curr_obj_pos))
          if dist < self.threshold and dist < self._dist_of_cur_held_obj:
            self.attached_object = k
            self._dist_of_cur_held_obj = dist
    else:
      self.attached_object = (-1, -1)

    next_fist_pos = current_fist_pos + curr_action
    next_fist_pos = np.clip(next_fist_pos, -2.8, 2.8)
    if self.attached_object != (-1, -1):
      current_obj_pos = np.array([
          self.sim.data.qpos[i] for i in self.object_dict[self.attached_object]
      ])
      current_obj_pos += (next_fist_pos - current_fist_pos)
      current_obj_pos = np.clip(current_obj_pos, -2.8, 2.8)

    # Setting the final positions
    curr_qpos = self.sim.data.qpos.copy()
    curr_qpos[0] = next_fist_pos[0]
    curr_qpos[1] = next_fist_pos[1]
    if self.attached_object != (-1, -1):
      for enum_n, i in enumerate(self.object_dict[self.attached_object]):
        curr_qpos[i] = current_obj_pos[enum_n]

    self.set_state(curr_qpos)
    self.sim.forward()

  def _compute_reward_old(self, obs):
    if self._reward_type == "sparse":
      reward = float(self.is_successful(obs=obs))
    elif self._reward_type == "dense":
      # remove gripper, attached object from reward computation
      reward = -np.linalg.norm(obs[2:4] - obs[8:-2])
      for obj_idx in range(1, 2):
        reward += 2. * np.exp(
            -(np.linalg.norm(obs[2 * obj_idx:2 * obj_idx + 2] -
                             obs[2 * obj_idx + 6:2 * obj_idx + 8])**2) / 0.01)

      grip_to_object = 0.5 * np.linalg.norm(obs[:2] - obs[2:4])
      reward += -grip_to_object
      reward += 0.5 * np.exp(-(grip_to_object**2) / 0.01)

    return reward

  def compute_reward(self, obs, vectorized=True):
    if self._reward_type == "sparse":
      reward = np.array(self.is_successful(obs=obs), dtype=np.float32)
    elif self._reward_type == "dense":
      # remove gripper, attached object from reward computation
      obs = np.atleast_2d(obs)
      reward = -np.linalg.norm(obs[:,2:4] - obs[:,8:-2], axis=-1)
      for obj_idx in range(1, 2):
        reward += 2. * np.exp(
            -(np.linalg.norm(obs[:,2 * obj_idx:2 * obj_idx + 2] -
                             obs[:,2 * obj_idx + 6:2 * obj_idx + 8], axis=-1)**2) / 0.01)

      grip_to_object = 0.5 * np.linalg.norm(obs[:,:2] - obs[:,2:4], axis=-1)
      reward += -grip_to_object
      reward += 0.5 * np.exp(-(grip_to_object**2) / 0.01)

    return np.squeeze(reward)

  def get_obs(self):
    return self._get_obs()

  def is_successful(self, obs=None, vectorized=True):
    return self.dist_to_goal(obs=obs, vectorized=vectorized) <= 0.2

  # 0:2 gripper, 2:4 mug, 6:8 goal gripper, 8:10 goal mug pos
  def dist_to_goal(self, obs=None, vectorized=True):
    if obs is None:
      obs = self._get_obs()

    if self._wide_init_distr:
      return np.linalg.norm(obs[:,2:4] - obs[:,8:-2], axis=-1) if obs.ndim == 2 else np.linalg.norm(obs[2:4] - obs[8:-2])
    else:
      return np.linalg.norm(obs[:,:4] - obs[:,6:-2], axis=-1) if obs.ndim == 2 else np.linalg.norm(obs[:4] - obs[6:-2])
  

class TabletopManipulationImage(TabletopManipulation):
  def __init__(self, camera_name="camera1"):
    self.reset_atleast_once = False
    self._camera_name = camera_name
    super(TabletopManipulationImage, self).__init__(
      task_list="rc_r-rc_k-rc_g-rc_b",
      reward_type="dense",
      reset_at_goal=False)
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8,
    )
    self.observation_space = self._new_observation_space

  def reset(self):
    self.reset_atleast_once = True

    # generate new goal image
    # self.observation_space = self._old_observation_space
    self._reset_at_goal = True
    s = super(TabletopManipulationImage, self).reset()
    # self.observation_space = self._new_observation_space
    self._reset_at_goal = False
    self._goal_img = self.observation(s)

    # initial state image
    # self.observation_space = self._old_observation_space
    s = super(TabletopManipulationImage, self).reset()
    # self.observation_space = self._new_observation_space
    img = self.observation(s)

    return {'observation': img, 'desired_goal': self._goal_img}
  
  def step(self, action):
    if self.reset_atleast_once:
      s, _, _, _ = super(TabletopManipulationImage, self).step(action)
      done = False
      r = self.compute_reward(s)
      info = {'is_success': self.is_successful(obs=s)}
      img = self.observation(s)
      return {'observation': img, 'desired_goal': self._goal_img}, r, done, info
    else: # workaround for MujocoEnv initialization
      return super(TabletopManipulationImage, self).step(action)

  def observation(self, observation):
    img = self.render(mode='rgb_array', height=64, width=64)
    return img.transpose(2,0,1) # HWC -> CHW

  def viewer_setup(self):
    super(TabletopManipulationImage, self).viewer_setup()
    if self._camera_name == 'original':
      pass
    elif self._camera_name == 'camera1':
      self.viewer.cam.lookat[1] = 0.0
      self.viewer.cam.distance = 10.5
    elif self._camera_name == 'camera2':
      self.viewer.cam.distance = 10
      self.viewer.cam.elevation = -90
    else:
      raise NotImplementedError