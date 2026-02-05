import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN
from almacen_1 import WarehouseEnv

# Cargar modelo entrenado
entorno = "1"
entornos = {"1": {"random_objects": False, "just_pick": True, "render_mode": "human"},
            "2": {"random_objects": False, "just_pick": False, "render_mode": "human"},
            "3": {"random_objects": True,  "just_pick": False, "render_mode": "human"}}
env = WarehouseEnv(**entornos[entorno])
model = DQN.load(f"dqn_warehouse_weights_entorno_{entorno}", env=env)


def evaluate_agent(model, env, num_episodes=500, render=False):
    rewards_list = []
    success_list = []
    number_steps = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated
            last_info = info  # guardamos el info final

            if render:
                env.render()

        rewards_list.append(total_reward)

        # -------------------------
        #    CÁLCULO DEL ÉXITO
        # -------------------------
        if env.just_pick:
            # ENTORNO 1 — SOLO RECOGER
            collision = bool(obs[9])
            picked = bool(obs[8])       # flag de que tiene objeto en el último frame
            success = 1 if (picked and not collision and not truncated) else 0

        else:
            # ENTORNOS 2 Y 3 — RECOGER Y DEJAR
            success = 1 if last_info.get("success", False) else 0

        success_list.append(success)
        number_steps.append(steps)

    rewards_array = np.array(rewards_list)
    success_rate = np.cumsum(success_list) / np.arange(1, num_episodes + 1) * 100

    # -------------------------
    #       GRAFICAS
    # -------------------------
    plt.figure(figsize=(15, 5))

    # Gráfica 1: Recompensas
    plt.subplot(1, 3, 1)
    colors = ['green' if s == 1 else 'red' for s in success_list]
    plt.scatter(np.arange(1, num_episodes+1), rewards_array, s=10, c=colors)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.title("Recompensas por episodio")
    plt.grid(True)

    # Gráfica 2: Tasa de éxito acumulada
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(1, num_episodes+1), success_rate)
    plt.xlabel("Episodio")
    plt.ylabel("% Episodios exitosos")
    plt.title("Tasa de éxito acumulada")
    plt.ylim([0, 100])
    plt.grid(True)

    # Gráfica 3: Steps por episodio
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(1, num_episodes+1), number_steps)
    plt.xlabel("Episodio")
    plt.ylabel("Steps")
    plt.title("Steps por episodio")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"dqn_evaluation_{entorno}.png", dpi=300)
    plt.show()

    print(f"Recompensa promedio: {rewards_array.mean():.2f}")
    print(f"Tasa de éxito promedio: {success_rate[-1]}%")
    print(f"Steps promedio: {np.mean(number_steps):.2f}")

    return rewards_array, success_rate


# Evaluar
rewards, success = evaluate_agent(model, env, num_episodes=100, render=True)
env.close()