from sac.envs.ant_maze_env import AntMazeEnv
from sac.envs.point_mass import PointMassEnv
from gym.spaces import Box
import gym
import numpy as np


REWARD_SCALE = 0.1
DISTANCE_THRESHOLD = 5


class AntMaze(AntMazeEnv):
    """Universal environment variant of AntMazeEnv.
    FIXME
    This environment extends the generic gym environment by including contexts,
    or goals. The goals are added to the observation, and an additional
    contextual reward is included to the generic rewards. If a certain goal is
    met, the environment registers a "done" flag and the environment is reset.
    """

    def __init__(self, fixed_goal=None):
        """Initialize the Universal environment.
        Parameters
        ----------
        fixed_goal : array_like
            if provided sets the environment goal to this
        """

        self.wrapped_env = AntMazeEnv(
            maze_id="Maze",
            maze_height=0.5,
            maze_size_scaling=8,
            n_bins=0,
            sensor_range=3.,
            sensor_span=2 * np.pi,
            observe_blocks=False,
            put_spin_near_agent=False,
            top_down_view=False,
            image_size=32,
            manual_collision=False)

        self.fixed_goal = fixed_goal
        self.horizon = 500
        self.step_number = 0
        self.current_context = None
        self.context_space = Box(low=np.array([-4.0, -4.0]),
                                 high=np.array([12.0, 12.0]),
                                 dtype=np.float32)

    def step(self, action):
        """Advance the environment by one simulation step.
        If the environment is using the contextual setting, an "is_success"
        term is added to the info_dict to specify whether the objective has
        been met.
        Parameters
        ----------
        action : array_like
            actions to be performed by the agent
        Returns
        -------
        array_like
            next observation
        float
            environmental reward
        bool
            done mask
        dict
            extra information dictionary
        """

        obs, rew, done, info = self.wrapped_env.step(action)
        rew = -np.linalg.norm(
            self.current_context - obs, ord=2) * REWARD_SCALE

        info["is_success"] = abs(rew) < DISTANCE_THRESHOLD * REWARD_SCALE
        self.step_number += 1
        done = done or self.step_number == self.horizon
        return obs, self.current_context - obs, rew, done, info

    def reset(self):
        """Reset the environment.
        If the environment is using the contextual setting, a new context is
        issued.
        Returns
        -------
        array_like
            initial observation
        """

        self.step_number = 0
        obs = self.wrapped_env.reset()
        self.current_context = self.context_space.sample()
        if self.fixed_goal is not None:
            self.current_context = self.fixed_goal
        return obs, self.current_context - obs


class PointMass(gym.core.Wrapper):
    """Universal environment variant of PointMass.
    FIXME
    This environment extends the generic gym environment by including contexts,
    or goals. The goals are added to the observation, and an additional
    contextual reward is included to the generic rewards. If a certain goal is
    met, the environment registers a "done" flag and the environment is reset.
    """

    def __init__(self, fixed_goal=None):
        """Initialize the Universal environment.
        Parameters
        ----------
        fixed_goal : array_like
            if provided sets the environment goal to this
        """

        super(PointMass, self).__init__(PointMassEnv())

        self.fixed_goal = fixed_goal
        self.horizon = 50
        self.step_number = 0

        self.current_context = None
        self.context_space = Box(
            low=np.array([-2.0, 2.0], dtype=np.float32),
            high=np.array([-2.0, 2.0], dtype=np.float32), dtype=np.float32)

    def step(self, action):
        """Advance the environment by one simulation step.
        If the environment is using the contextual setting, an "is_success"
        term is added to the info_dict to specify whether the objective has
        been met.
        Parameters
        ----------
        action : array_like
            actions to be performed by the agent
        Returns
        -------
        array_like
            next observation
        float
            environmental reward
        bool
            done mask
        dict
            extra information dictionary
        """

        obs, rew, done, info = self.env.step(action)
        rew = -np.linalg.norm(
            self.current_context - obs[:2], ord=2) * REWARD_SCALE

        info["is_success"] = abs(rew) < DISTANCE_THRESHOLD * REWARD_SCALE
        self.step_number += 1
        done = done or self.step_number == self.horizon
        return obs, self.current_context - obs[:2], rew, done, info

    def reset(self):
        """Reset the environment.
        If the environment is using the contextual setting, a new context is
        issued.
        Returns
        -------
        array_like
            initial observation
        """

        self.step_number = 0
        obs = self.env.reset()
        self.current_context = self.context_space.sample()
        if self.fixed_goal is not None:
            self.current_context = self.fixed_goal
        return obs, self.current_context - obs[:2]
