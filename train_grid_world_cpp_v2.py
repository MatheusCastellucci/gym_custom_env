#
# Curriculum Learning for Coverage Path Planning - V2
#
# Training strategy: 5x5 (Phase 1) -> 10x10 (Phase 2) -> 10x10 fine-tune (Phase 3).
# The model generalizes to 20x20 without any 20x20 training (zero-shot generalization).
#
# Usage:
#   python train_grid_world_cpp_v2.py train   -> full curriculum from scratch (5x5 -> 10x10 -> phase3)
#   python train_grid_world_cpp_v2.py test    -> test model on 5x5, 10x10, and 20x20
#   python train_grid_world_cpp_v2.py run     -> visualize model (asks for model and size)
#

import sys
import numpy as np
import gymnasium as gym
from datetime import datetime

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# ---------------------------------------------------------------------------
# Environment registration
# ---------------------------------------------------------------------------
try:
    gym.register(
        id="gymnasium_env/GridWorldCPPV2-v0",
        entry_point=GridWorldCPPEnvV2,
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Phase 1: small grid — learn basic coverage behavior
P1_DIM        = 5
P1_OBSTACLES  = 3
P1_MAX_STEPS  = 200
P1_TIMESTEPS  = 1_000_000
P1_GAMMA      = 0.99
P1_LR         = 3e-4
P1_ENT_COEF   = 0.05

# Phase 2: target grid — transfer from 5x5
P2_DIM        = 10
P2_OBSTACLES  = 12
P2_MAX_STEPS  = 400
P2_TIMESTEPS  = 1_000_000
P2_GAMMA      = 0.99
P2_LR         = 3e-4
P2_ENT_COEF   = 0.05

# Phase 3: fine-tune for last-mile completion on 10x10
# Longer episodes + higher gamma so the agent values completing distant last cells.
# Lower LR and entropy for conservative exploitation of the learned policy.
P3_DIM        = 10
P3_OBSTACLES  = 12
P3_MAX_STEPS  = 600   # extra headroom for last-cell navigation
P3_TIMESTEPS  = 500_000
P3_GAMMA      = 0.995  # longer horizon — last cell bonus visible even 200+ steps ahead
P3_LR         = 1e-4   # conservative update to avoid forgetting Phase 2 behavior
P3_ENT_COEF   = 0.02   # less exploration; exploit the learned navigation

# Phase 4: scale to 20x20
# Obstacle ratio kept at ~12% (same as 5x5 and 10x10): 400 * 0.12 = 48.
# max_steps scaled to give ~5x the minimum path (352 free cells * 5 ≈ 1800).
# gamma=0.999: with 1000-step episodes, 10*0.999^800 ≈ 4.5 vs 10*0.995^800 ≈ 0.018.
# Higher timesteps because each episode is much longer (slower gradient updates).
P4_DIM        = 20
P4_OBSTACLES  = 48
P4_MAX_STEPS  = 2000
P4_TIMESTEPS  = 2_000_000
P4_GAMMA      = 0.999
P4_LR         = 1e-4
P4_ENT_COEF   = 0.02

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_name(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "?")


def make_env(size, obstacles, max_steps, render_mode="rgb_array"):
    return gym.make(
        "gymnasium_env/GridWorldCPPV2-v0",
        size=size,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode=render_mode,
    )


def train_phase(
    dim, obstacles, max_steps, timesteps, label,
    pretrained_path=None, reset_timesteps=True,
    gamma=0.99, learning_rate=3e-4, ent_coef=0.05,
):
    """Train (or fine-tune) a PPO model and return (model, saved_path)."""
    env = make_env(dim, obstacles, max_steps)
    check_env(env)

    if pretrained_path:
        print(f"[{label}] Loading pretrained model: {pretrained_path}")
        model = PPO.load(
            pretrained_path, env=env,
            # Override hyperparameters for this phase
            custom_objects={
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "gamma": gamma,
            },
        )
    else:
        model = PPO(
            "MultiInputPolicy", env,
            verbose=1,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            device="cpu",
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = f"log/ppo_cpp_v2_{dim}x{dim}_{label}_{ts}"
    model_path = f"data/ppo_cpp_v2_{dim}x{dim}_{label}_{ts}"

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    print(f"[{label}] Training {timesteps:,} timesteps on {dim}x{dim} | "
          f"max_steps={max_steps} gamma={gamma} lr={learning_rate} ent_coef={ent_coef}")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset_timesteps)
    model.save(model_path)
    print(f"[{label}] Model saved -> {model_path}.zip")
    print(f"[{label}] Logs   saved -> {log_dir}")
    return model, model_path


def run_test(model, dim, obstacles, max_steps, num_episodes=100):
    """Run evaluation episodes and print statistics."""
    env = make_env(dim, obstacles, max_steps)

    coverages = []
    steps_list = []
    full_coverage_count = 0

    for i in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, truncated, info = env.step(action.item())
            steps += 1

        coverages.append(info["coverage"])
        steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"  Episode {i+1:>3}: FULL coverage in {steps} steps")
        else:
            print(f"  Episode {i+1:>3}: {info['coverage']:.1%} coverage in {steps} steps")

    print(f"\n--- Test summary ({dim}x{dim}, max_steps={max_steps}) ---")
    print(f"Full Coverage Rate : {full_coverage_count}/{num_episodes} "
          f"({full_coverage_count/num_episodes*100:.1f}%)")
    print(f"Avg Coverage       : {np.mean(coverages)*100:.1f}% "
          f"(std={np.std(coverages)*100:.1f}%,  "
          f"min={np.min(coverages)*100:.1f}%,  max={np.max(coverages)*100:.1f}%)")
    print(f"Avg Steps          : {np.mean(steps_list):.1f} "
          f"(std={np.std(steps_list):.1f},  "
          f"min={np.min(steps_list)},  max={np.max(steps_list)})")
    return full_coverage_count, coverages, steps_list


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

VALID_MODES = {"train", "test", "run"}

if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
    print(f"Usage: python {sys.argv[0]} <{'|'.join(sorted(VALID_MODES))}>")
    sys.exit(1)

mode = sys.argv[1]

# ---- FULL CURRICULUM (1 → 2 → 3) ------------------------------------------
if mode == "train":
    print("=" * 60)
    print("CURRICULUM: Phase 1 (5x5) -> Phase 2 (10x10) -> Phase 3 (10x10 fine-tune)")
    print("The resulting model generalizes to 20x20 without any 20x20 training.")
    print("=" * 60)

    _, phase1_path = train_phase(
        P1_DIM, P1_OBSTACLES, P1_MAX_STEPS, P1_TIMESTEPS,
        label="phase1", reset_timesteps=True,
        gamma=P1_GAMMA, learning_rate=P1_LR, ent_coef=P1_ENT_COEF,
    )

    print("\n--- Phase 1 done. Starting Phase 2 ---\n")
    _, phase2_path = train_phase(
        P2_DIM, P2_OBSTACLES, P2_MAX_STEPS, P2_TIMESTEPS,
        label="phase2",
        pretrained_path=phase1_path + ".zip",
        reset_timesteps=False,
        gamma=P2_GAMMA, learning_rate=P2_LR, ent_coef=P2_ENT_COEF,
    )

    print("\n--- Phase 2 done. Starting Phase 3 (last-mile fine-tune on 10x10) ---\n")
    train_phase(
        P3_DIM, P3_OBSTACLES, P3_MAX_STEPS, P3_TIMESTEPS,
        label="phase3",
        pretrained_path=phase2_path + ".zip",
        reset_timesteps=False,
        gamma=P3_GAMMA, learning_rate=P3_LR, ent_coef=P3_ENT_COEF,
    )

# ---- TEST ------------------------------------------------------------------
elif mode == "test":
    model_name = input("Enter model filename (without .zip, from data/): ").strip()
    model = PPO.load(f"data/{model_name}.zip")

    sizes = [
        (5,  P1_OBSTACLES, P1_MAX_STEPS),
        (10, P3_OBSTACLES, P3_MAX_STEPS),
        (20, P4_OBSTACLES, P4_MAX_STEPS),
    ]
    for dim, obs, ms in sizes:
        print(f"\n{'='*50}")
        print(f"Testing on {dim}x{dim} grid ({obs} obstacles, max {ms} steps)")
        print("="*50)
        run_test(model, dim, obs, ms, num_episodes=100)

# ---- RUN (VISUALIZE) -------------------------------------------------------
elif mode == "run":
    model_name = input("Enter model filename (without .zip, from data/): ").strip()
    model = PPO.load(f"data/{model_name}.zip")

    size_str = input("Grid size to visualize [5/10/20] (default 10): ").strip() or "10"
    dim = int(size_str)
    _cfg = {5: (P1_OBSTACLES, P1_MAX_STEPS), 10: (P3_OBSTACLES, P3_MAX_STEPS), 20: (P4_OBSTACLES, P4_MAX_STEPS)}
    obstacles, max_steps = _cfg.get(dim, (P4_OBSTACLES, P4_MAX_STEPS))

    env = make_env(dim, obstacles, max_steps, render_mode="human")
    obs, info = env.reset()
    done = truncated = False
    steps = 0
    total_reward = 0.0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1
        print(f"Step {steps:>4} | Action: {_action_name(action.item()):<5} | "
              f"Reward: {reward:+.2f} | Coverage: {info['coverage']:.1%}")

    print(f"\nFinished: coverage={info['coverage']:.1%}, "
          f"steps={steps}, total_reward={total_reward:.2f}")
    env.close()
