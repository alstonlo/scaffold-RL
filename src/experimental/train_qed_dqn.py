import copy
import pathlib
import statistics

import torch
import torch.nn.functional as F
import wandb

from src.agents import DQNAgent
from src.dqn import ScaffoldDQN
from src.environments import QEDScaffoldDecorator
from src.utils.replay_buffer import ReplayBuffer
from src.utils.seed_utils import seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wandb_checkpoint(model):
    model_path = str(pathlib.Path(wandb.run.dir) / "model.pt")
    torch.save(model, model_path)
    wandb.save(model_path)


def target_dqn_update(dqn, target_dqn, polyak):
    with torch.no_grad():
        for p, target_p in zip(dqn.parameters(), target_dqn.parameters()):
            target_p.data.mul_(polyak)
            target_p.data.add_((1 - polyak) * p.data)


def dqn_update(dqn, target_dqn, batch, optimizer):
    dqn.train()

    sa_ts, rewards, sa_tp1ses, dones = batch

    rewards = torch.tensor(rewards, dtype=torch.float).to(DEVICE)
    v_tp1s = torch.zeros_like(rewards)
    for i, sa_tp1s in enumerate(sa_tp1ses):
        if not dones[i]:
            v_tp1s[i] = torch.max(target_dqn(sa_tp1s))
    td_target = (rewards + v_tp1s)
    q_ts = dqn(sa_ts).squeeze(1)
    loss = F.huber_loss(td_target, q_ts)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dqn.eval()
    return loss.item()


def train_double_dqn(
        dqn, env, buffer_size,
        n_episodes, batch_size, lr,
        learn_freq, update_freq, polyak,
):
    assert env.max_steps % learn_freq == 0
    assert env.max_steps % update_freq == 0
    dqn.eval()

    agent = DQNAgent(dqn, epsilon=1.0)
    eps_anneal = 0.01 ** (1 / n_episodes)

    replay_buffer = ReplayBuffer(buffer_size)
    target_dqn = copy.deepcopy(dqn).to(DEVICE)
    for p in target_dqn.parameters():
        p.requires_grad = False
    target_dqn.eval()

    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)

    for episode in range(n_episodes):
        env.reset()
        losses = []
        value = 0.0

        for step in range(env.max_steps):

            with torch.no_grad():
                act = agent.sample_action(env)
            next_obs, reward, done = env.step(act)
            value += reward

            # since MDP is deterministic (s, a) can be represented with s'
            sa_t = next_obs
            sa_tp1s = [(a, env.state[1] - 1) for a in env.valid_actions]
            replay_buffer.add(sa_t, reward, sa_tp1s, done)

            # perform double DQN update
            if (step + 1) % learn_freq == 0:
                batch = replay_buffer.sample(batch_size)
                loss = dqn_update(dqn, target_dqn, batch, optimizer)
                losses.append(loss)
            if (step + 1) % update_freq == 0:
                target_dqn_update(dqn, target_dqn, polyak)

        agent.epsilon *= eps_anneal

        # wandb logging
        avg_loss = statistics.mean(losses)
        qed = env.prop_fn(env.state[0])

        wandb.log({"Episode": episode, "Value": value, "QED": qed, "Loss": avg_loss})
        if episode % 100 == 0:
            wandb_checkpoint(model=dqn)


def main():
    log_dir = pathlib.Path(__file__).parents[2] / "logs" / "qed_dqn"
    log_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="opt_QED_DQN", tensorboard=True, dir=log_dir)

    env = QEDScaffoldDecorator(init_mol="[CH3:1][CH3:1]")

    seed_everything(seed=498)
    dqn = ScaffoldDQN(
        atom_types=env.atom_types,
        device=DEVICE,
        num_layers=5,
        emb_dim=300,
        dropout=0.5,
    )

    seed_everything(seed=498)
    train_double_dqn(
        dqn=dqn, env=env, buffer_size=5000,
        n_episodes=2000, batch_size=16, lr=1e-4,
        learn_freq=4, update_freq=20, polyak=0.995
    )


if __name__ == "__main__":
    main()