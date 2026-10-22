# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "mujoco_cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_ids: tuple[str, ...] = (
        "HalfCheetah-v5",
        "Walker2d-v5",
        "Ant-v5",
        "Humanoid-v5",
        "HalfCheetah-v5",
        "Walker2d-v5",
        "Ant-v5",
        "Humanoid-v5",
        "HalfCheetah-v5",
        "Walker2d-v5",
        "Ant-v5",
        "Humanoid-v5",
    )
    """the MuJoCo tasks trained sequentially (reversed 4-task order repeated three times)"""
    total_timesteps: int = 1000000
    """timesteps for EACH task in the sequence"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    use_without_clip: bool = False
    """Toggles whether or not to use the unclipped surrogate loss for the policy."""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    trust_type: Literal[
        "none",
        "pearson",
        "k3",
        "gaussian",
        "mixture_gaussian",
    ] = "none"
    """trust-term type for policy optimization"""
    model_arch: Literal["mlp", "simba"] = "simba"
    """policy/value network architecture"""
    weight_decay: float = 0.0
    """weight decay used in Adam optimizer"""
    mlp_hidden_dim: int = 64
    """hidden dimension for the baseline MLP architecture"""
    simba_actor_hidden_dim: int = 128
    """base actor hidden dimension for SimBa"""
    simba_actor_num_blocks: int = 1
    """number of SimBa residual blocks for actor"""
    simba_critic_hidden_dim: int = 512
    """base critic hidden dimension for SimBa"""
    simba_critic_num_blocks: int = 2
    """number of SimBa residual blocks for critic"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations per task (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def infer_env_dims(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    obs_dim = int(np.array(env.observation_space.shape).prod())
    act_dim = int(np.prod(env.action_space.shape))
    env.close()
    return obs_dim, act_dim


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def he_layer_init(layer, bias_const=0.0):
    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared across tasks.
        self.critic_shared = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_shared_middle = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Task-specific modules.
        self.value_encoders = nn.ModuleDict()
        self.actor_encoders = nn.ModuleDict()
        self.actor_heads = nn.ModuleDict()
        self.actor_logstd = nn.ParameterDict()
        self._task_dims: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _sanitize_task_key(task_id: str) -> str:
        return task_id.replace("/", "__")

    def register_task(self, task_id: str, obs_dim: int, action_dim: int) -> str:
        task_key = self._sanitize_task_key(task_id)
        if task_key in self._task_dims:
            if self._task_dims[task_key] != (obs_dim, action_dim):
                raise ValueError(
                    f"Task {task_id} was already registered with dims={self._task_dims[task_key]}, "
                    f"new dims={(obs_dim, action_dim)}"
                )
            return task_key

        ref_param = next(self.critic_shared.parameters())
        module_device = ref_param.device
        module_dtype = ref_param.dtype

        self.value_encoders[task_key] = layer_init(nn.Linear(obs_dim, self.hidden_dim)).to(
            device=module_device, dtype=module_dtype
        )
        self.actor_encoders[task_key] = layer_init(nn.Linear(obs_dim, self.hidden_dim)).to(
            device=module_device, dtype=module_dtype
        )
        self.actor_heads[task_key] = layer_init(nn.Linear(self.hidden_dim, action_dim), std=0.01).to(
            device=module_device, dtype=module_dtype
        )
        self.actor_logstd[task_key] = nn.Parameter(torch.zeros(1, action_dim, device=module_device, dtype=module_dtype))
        self._task_dims[task_key] = (obs_dim, action_dim)
        return task_key

    def shared_parameters(self):
        return list(self.critic_shared.parameters()) + list(self.actor_shared_middle.parameters())

    def task_parameters(self, task_id: str):
        task_key = self._sanitize_task_key(task_id)
        if task_key not in self._task_dims:
            raise KeyError(f"Task not registered: {task_id}")
        return [
            *list(self.value_encoders[task_key].parameters()),
            *list(self.actor_encoders[task_key].parameters()),
            *list(self.actor_heads[task_key].parameters()),
            self.actor_logstd[task_key],
        ]

    def get_value(self, x, task_id: str):
        task_key = self._sanitize_task_key(task_id)
        value_hidden = torch.tanh(self.value_encoders[task_key](x))
        return self.critic_shared(value_hidden)

    def get_action_and_value(self, x, task_id: str, action=None):
        task_key = self._sanitize_task_key(task_id)
        actor_hidden = torch.tanh(self.actor_encoders[task_key](x))
        actor_hidden = self.actor_shared_middle(actor_hidden)
        action_mean = self.actor_heads[task_key](actor_hidden)
        action_logstd = self.actor_logstd[task_key].expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        return action, log_prob, entropy, self.get_value(x, task_id), action_mean, action_std


class SimBaResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = he_layer_init(nn.Linear(hidden_dim, hidden_dim * 4))
        self.fc2 = he_layer_init(nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class SimBaAgent(nn.Module):
    def __init__(
        self,
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 1,
        critic_hidden_dim: int = 512,
        critic_num_blocks: int = 2,
        actor_head_std: float = 0.01,
    ):
        super().__init__()
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor_head_std = actor_head_std

        # Shared across tasks.
        self.actor_blocks = nn.ModuleList(
            [SimBaResidualBlock(self.actor_hidden_dim) for _ in range(actor_num_blocks)]
        )
        self.actor_post_norm = nn.LayerNorm(self.actor_hidden_dim)
        self.critic_blocks = nn.ModuleList(
            [SimBaResidualBlock(self.critic_hidden_dim) for _ in range(critic_num_blocks)]
        )
        self.critic_post_norm = nn.LayerNorm(self.critic_hidden_dim)
        self.critic_head = layer_init(nn.Linear(self.critic_hidden_dim, 1), std=1.0)

        # Task-specific modules.
        self.actor_input_layers = nn.ModuleDict()
        self.critic_input_layers = nn.ModuleDict()
        self.actor_heads = nn.ModuleDict()
        self.actor_logstd = nn.ParameterDict()
        self._task_dims: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _sanitize_task_key(task_id: str) -> str:
        return task_id.replace("/", "__")

    def register_task(self, task_id: str, obs_dim: int, action_dim: int) -> str:
        task_key = self._sanitize_task_key(task_id)
        if task_key in self._task_dims:
            if self._task_dims[task_key] != (obs_dim, action_dim):
                raise ValueError(
                    f"Task {task_id} was already registered with dims={self._task_dims[task_key]}, "
                    f"new dims={(obs_dim, action_dim)}"
                )
            return task_key

        ref_param = next(self.critic_head.parameters())
        module_device = ref_param.device
        module_dtype = ref_param.dtype

        self.actor_input_layers[task_key] = layer_init(nn.Linear(obs_dim, self.actor_hidden_dim), std=1.0).to(
            device=module_device, dtype=module_dtype
        )
        self.critic_input_layers[task_key] = layer_init(nn.Linear(obs_dim, self.critic_hidden_dim), std=1.0).to(
            device=module_device, dtype=module_dtype
        )
        self.actor_heads[task_key] = layer_init(nn.Linear(self.actor_hidden_dim, action_dim), std=self.actor_head_std).to(
            device=module_device, dtype=module_dtype
        )
        self.actor_logstd[task_key] = nn.Parameter(torch.zeros(1, action_dim, device=module_device, dtype=module_dtype))
        self._task_dims[task_key] = (obs_dim, action_dim)
        return task_key

    def shared_parameters(self):
        return [
            *list(self.actor_blocks.parameters()),
            *list(self.actor_post_norm.parameters()),
            *list(self.critic_blocks.parameters()),
            *list(self.critic_post_norm.parameters()),
            *list(self.critic_head.parameters()),
        ]

    def task_parameters(self, task_id: str):
        task_key = self._sanitize_task_key(task_id)
        if task_key not in self._task_dims:
            raise KeyError(f"Task not registered: {task_id}")
        return [
            *list(self.actor_input_layers[task_key].parameters()),
            *list(self.critic_input_layers[task_key].parameters()),
            *list(self.actor_heads[task_key].parameters()),
            self.actor_logstd[task_key],
        ]

    def _actor_features(self, x: torch.Tensor, task_key: str) -> torch.Tensor:
        x = self.actor_input_layers[task_key](x)
        for block in self.actor_blocks:
            x = block(x)
        return self.actor_post_norm(x)

    def _critic_features(self, x: torch.Tensor, task_key: str) -> torch.Tensor:
        x = self.critic_input_layers[task_key](x)
        for block in self.critic_blocks:
            x = block(x)
        return self.critic_post_norm(x)

    def get_value(self, x, task_id: str):
        task_key = self._sanitize_task_key(task_id)
        value_features = self._critic_features(x, task_key)
        return self.critic_head(value_features)

    def get_action_and_value(self, x, task_id: str, action=None):
        task_key = self._sanitize_task_key(task_id)
        actor_features = self._actor_features(x, task_key)
        action_mean = self.actor_heads[task_key](actor_features)
        action_logstd = self.actor_logstd[task_key].expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic_head(self._critic_features(x, task_key))
        return action, log_prob, entropy, value, action_mean, action_std


def build_agent(args: Args) -> nn.Module:
    if args.model_arch == "simba":
        return SimBaAgent(
            actor_hidden_dim=args.simba_actor_hidden_dim,
            actor_num_blocks=args.simba_actor_num_blocks,
            critic_hidden_dim=args.simba_critic_hidden_dim,
            critic_num_blocks=args.simba_critic_num_blocks,
        )
    return MLPAgent(hidden_dim=args.mlp_hidden_dim)


def evaluate(
    model_path: str,
    make_env_fn,
    env_id: str,
    task_env_dims: dict[str, tuple[int, int]],
    eval_episodes: int,
    run_name: str,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    agent_args: Args | None = None,
):
    envs = gym.vector.SyncVectorEnv([make_env_fn(env_id, 0, capture_video, run_name, gamma)])
    if agent_args is None:
        agent = SimBaAgent().to(device)
    else:
        agent = build_agent(agent_args).to(device)
    for task_name, (obs_dim, action_dim) in task_env_dims.items():
        agent.register_task(task_name, obs_dim, action_dim)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _, _, _ = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32, device=device),
            task_id=env_id,
        )
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())

        if "episode" in infos:
            ep_info = infos["episode"]
            ep_returns = np.asarray(ep_info["r"], dtype=np.float32).reshape(-1)
            done_mask = np.asarray(infos["_episode"], dtype=bool).reshape(-1)

            for env_idx, is_done in enumerate(done_mask):
                if not is_done or env_idx >= ep_returns.shape[0]:
                    continue
                ep_ret = float(ep_returns[env_idx])
                print(f"eval_episode={len(episodic_returns)}, episodic_return={ep_ret}")
                episodic_returns += [ep_ret]
                if len(episodic_returns) >= eval_episodes:
                    break

        obs = next_obs

    envs.close()
    return episodic_returns


if __name__ == "__main__":
    args = tyro.cli(Args)
    if len(args.env_ids) == 0:
        raise ValueError("env_ids must contain at least one task")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps is too small for current batch_size")

    tasks_tag = "-".join([task.split("/")[-1] for task in args.env_ids])
    run_name = f"{tasks_tag}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_dims: dict[str, tuple[int, int]] = {}
    for env_id in args.env_ids:
        if env_id not in env_dims:
            env_dims[env_id] = infer_env_dims(env_id)

    agent = build_agent(args).to(device)
    optimizers: dict[str, optim.AdamW] = {}
    for env_id, (obs_dim, action_dim) in env_dims.items():
        agent.register_task(env_id, obs_dim, action_dim)
        task_params = [*agent.shared_parameters(), *agent.task_parameters(env_id)]
        optimizers[env_id] = optim.AdamW(task_params, lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay)

    optimizer_identity = {env_id: id(opt) for env_id, opt in optimizers.items()}
    task_visit_count = {env_id: 0 for env_id in env_dims}

    # TRY NOT TO MODIFY: global trackers across sequential tasks
    global_step = 0
    start_time = time.time()
    envs = None
    final_task_env_id = args.env_ids[-1]

    for task_id, task_env_id in enumerate(args.env_ids, start=1):
        if envs is not None:
            envs.close()
        envs = gym.vector.SyncVectorEnv(
            [make_env(task_env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        assert isinstance(envs.single_observation_space, gym.spaces.Box), "only box observation space is supported"

        task_obs_dim, task_action_dim = env_dims[task_env_id]
        optimizer = optimizers[task_env_id]
        if id(optimizer) != optimizer_identity[task_env_id]:
            raise RuntimeError(f"Optimizer object changed unexpectedly for task={task_env_id}")
        task_visit_count[task_env_id] += 1
        if task_visit_count[task_env_id] > 1:
            print(
                f"Revisiting task={task_env_id}: reusing optimizer id={id(optimizer)} "
                f"with state_entries={len(optimizer.state)}"
            )
        print(
            f"Starting task {task_id}/{len(args.env_ids)}: {task_env_id} "
            f"(obs_dim={task_obs_dim}, act_dim={task_action_dim})"
        )

        # ALGO Logic: Storage setup (task-specific dims)
        obs = torch.zeros((args.num_steps, args.num_envs, task_obs_dim), dtype=torch.float32, device=device)
        actions = torch.zeros((args.num_steps, args.num_envs, task_action_dim), dtype=torch.float32, device=device)
        logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

        # TRY NOT TO MODIFY: start the game
        next_obs_raw, _ = envs.reset(seed=args.seed + task_id - 1)
        next_obs = torch.tensor(next_obs_raw, dtype=torch.float32, device=device)
        next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _, _ = agent.get_action_and_value(next_obs, task_id=task_env_id)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done_np = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
                next_obs = torch.tensor(next_obs_raw, dtype=torch.float32, device=device)
                next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

                if "episode" in infos:
                    ep_info = infos["episode"]
                    ep_returns = np.asarray(ep_info["r"], dtype=np.float32).reshape(-1)
                    ep_lengths = np.asarray(ep_info["l"], dtype=np.int32).reshape(-1)
                    done_mask = np.asarray(infos["_episode"], dtype=bool).reshape(-1)

                    for env_idx, is_done in enumerate(done_mask):
                        if is_done and env_idx < ep_returns.shape[0] and env_idx < ep_lengths.shape[0]:
                            ep_ret = float(ep_returns[env_idx])
                            ep_len = int(ep_lengths[env_idx])
                            print(
                                f"task={task_env_id}, global_step={global_step}, "
                                f"episodic_return={ep_ret:.4f}"
                            )
                            writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                            writer.add_scalar("charts/episodic_length", ep_len, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs, task_id=task_env_id).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, task_obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, task_action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            epoch_old_logprob_bank = []
            stop_update = False

            old_approx_kl = torch.tensor(0.0, device=device)
            approx_kl = torch.tensor(0.0, device=device)
            pg_loss = torch.tensor(0.0, device=device)
            v_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
            trust_penalty_mean = torch.tensor(0.0, device=device)

            for epoch in range(args.update_epochs):
                mixture_old_logprobs = [b_logprobs] + epoch_old_logprob_bank
                mixture_old_logprobs_tensor = torch.stack(mixture_old_logprobs, dim=0)

                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        task_id=task_env_id,
                        action=b_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    if args.use_without_clip:
                        surrogate_loss = pg_loss1
                    else:
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        surrogate_loss = torch.max(pg_loss1, pg_loss2)

                    if args.trust_type == "none":
                        trust_penalty = torch.zeros_like(ratio)
                    elif args.trust_type == "pearson":
                        trust_penalty = 0.5 * (ratio - 1.0) ** 2
                    elif args.trust_type == "k3":
                        trust_penalty = ratio * logratio - ratio + 1.0
                    elif args.trust_type == "gaussian":
                        trust_penalty = 0.5 * (1.0 - torch.exp(-(ratio - 1.0) ** 2))
                    elif args.trust_type == "mixture_gaussian":
                        behavior_logprob = b_logprobs[mb_inds]
                        old_logprob_stack = mixture_old_logprobs_tensor[:, mb_inds]
                        ratio_stack = torch.exp(newlogprob.unsqueeze(0) - old_logprob_stack)
                        gaussian_stack = 0.5 * (1.0 - torch.exp(-(ratio_stack - 1.0) ** 2))
                        is_weight_stack = torch.exp(old_logprob_stack - behavior_logprob.unsqueeze(0))
                        trust_penalty = (is_weight_stack * gaussian_stack).mean(dim=0)
                    else:
                        raise ValueError(f"Unsupported trust_type: {args.trust_type}")

                    trust_weight = torch.abs(mb_advantages) / args.clip_coef
                    pg_loss = (surrogate_loss + trust_weight * trust_penalty).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    trust_penalty_mean = trust_penalty.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    trainable_params = [
                        p
                        for group in optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                    nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()

                with torch.no_grad():
                    _, epoch_logprob, _, _, _, _ = agent.get_action_and_value(
                        b_obs,
                        task_id=task_env_id,
                        action=b_actions,
                    )
                    epoch_old_logprob_bank.append(epoch_logprob.detach())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                if stop_update:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/trust_penalty", trust_penalty_mean.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print(f"task={task_env_id}, SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        final_task_env_id = task_env_id

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        can_run_default_eval = len(args.env_ids) == 1

        if can_run_default_eval:
            episodic_returns = evaluate(
                model_path,
                make_env,
                final_task_env_id,
                task_env_dims=env_dims,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                device=device,
                gamma=args.gamma,
                agent_args=args,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{final_task_env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
        else:
            print("Skipping default evaluate/push_to_hub because env_ids contains multiple tasks.")

    if envs is not None:
        envs.close()
    writer.close()
