import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
import os
from almacen_1 import WarehouseEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        # Log training metrics every 100 steps
        if self.n_calls % 100 == 0:
            # Log epsilon value (exploration rate)
            if hasattr(self.model, 'exploration_rate'):
                self.logger.record('train/epsilon', self.model.exploration_rate)
        
        # Evaluate the agent every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            mean_reward, success_rate, avg_steps = self._evaluate_model()
            
            # Log evaluation metrics
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/success_rate', success_rate)
            self.logger.record('eval/avg_steps', avg_steps)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"best_model_{self.n_calls}")
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")
        
        return True
    
    def _evaluate_model(self, num_episodes=10):
        """
        Evaluate the model and return metrics
        """
        rewards = []
        successes = []
        steps_list = []
        
        for ep in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            rewards.append(total_reward)
            success = 1 if (self.eval_env.delivery or (self.eval_env.just_pick and self.eval_env.agent_has_object)) and not self.eval_env.collision else 0
            successes.append(success)
            steps_list.append(steps)
        
        return np.mean(rewards), np.mean(successes) * 100, np.mean(steps_list)

class DQNAgent:
    def __init__(self, env, num_episodes, log_dir="./tensorboard_logs/"):
        self.env = env
        self.eval_env = WarehouseEnv(random_objects=self.env.random_objects,
                             just_pick=self.env.just_pick,
                             render_mode="human")
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=5e-4,
            buffer_size=20000,
            learning_starts=1000,
            batch_size=128,
            tau=0.1,
            gamma=0.95,
            train_freq=1,
            target_update_interval=5,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            policy_kwargs=dict(
                net_arch=[128, 128],
                activation_fn=torch.nn.ReLU
            ),
            verbose=1,
            tensorboard_log=log_dir,  # Enable tensorboard logging
        )
        self.num_episodes = num_episodes
        
    def train(self, entorno, total_timesteps=200000):
        print("=== Entrenamiento DQN con TensorBoard ===")
        print(f"Los logs de TensorBoard se guardan en: {self.log_dir}")
        
        # Create tensorboard callback
        tensorboard_callback = TensorboardCallback(
            eval_env=self.eval_env, 
            eval_freq=5000,  # Evaluate every 5000 steps
            verbose=1
        )
        
        # Train with callback
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=tensorboard_callback,
            tb_log_name=f"DQN_entorno_{entorno}",  # Unique name for this run
            log_interval=100  # Log training metrics every 100 steps
        )
        
        self.model.save(f"dqn_warehouse_weights_entorno_{entorno}")
        print(f"Pesos guardados en dqn_warehouse_weights_entorno_{entorno}")

    def evaluate(self, num_episodes=10):
        print("=== Evaluación DQN ===")
        rewards_list = []
        success_list = []
        steps_list = []

        for ep in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                self.eval_env.render()
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            rewards_list.append(total_reward)
            success = 1 if (self.eval_env.delivery or (self.eval_env.just_pick and self.eval_env.agent_has_object)) and not self.eval_env.collision else 0
            success_list.append(success)
            steps_list.append(steps)

        # Calculate metrics
        episodes = np.arange(1, num_episodes+1)
        success_rate = np.cumsum(success_list) / episodes * 100

        # Graficar
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        colors = ['green' if s == 1 else 'red' for s in success_list]
        plt.scatter(episodes, rewards_list, s=20, c=colors, alpha=0.7)
        plt.xlabel("Episodios")
        plt.ylabel("Recompensa Total")
        plt.title("Recompensas por episodio\n(Verde=Éxito, Rojo=Fallo)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(episodes, success_rate)
        plt.xlabel("Episodios")
        plt.ylabel("% Éxitos Acumulados")
        plt.title("Tasa de Éxito Acumulada")
        plt.ylim([0, 100])
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(episodes, steps_list, marker='o', alpha=0.7)
        plt.xlabel("Episodios")
        plt.ylabel("Pasos")
        plt.title("Pasos por Episodio")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"evaluation_entorno_{entorno}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary
        print(f"\nRESUMEN DE EVALUACIÓN:")
        print(f"Recompensa promedio: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
        print(f"Tasa de éxito final: {success_rate[-1]:.2f}%")
        print(f"Pasos promedio: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
        print(f"Éxitos: {np.sum(success_list)}/{num_episodes}")

        return rewards_list, success_rate, steps_list

    def log_custom_metrics(self, metrics_dict):
        """
        Log custom metrics to tensorboard
        """
        for key, value in metrics_dict.items():
            self.logger.record(key, value)

if __name__ == "__main__":
    entorno = "1"
    entornos = {
        "1": {"random_objects": False, "just_pick": True, "render_mode": "human"},
        "2": {"random_objects": False, "just_pick": False, "render_mode": "human"},
        "3": {"random_objects": True,  "just_pick": False, "render_mode": "human"}
    }
    
    # Create unique log directory for this run
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tensorboard_logs/entorno_{entorno}_{timestamp}/"
    
    env = WarehouseEnv(**entornos[entorno])
    agent = DQNAgent(env, num_episodes=10_000, log_dir=log_dir)
    
    print(f"Iniciando entrenamiento del Entorno {entorno}")
    print(f"Logs de TensorBoard: {log_dir}")
    
    agent.train(entorno=entorno, total_timesteps=200000)
    agent.evaluate(num_episodes=20)