import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, just_pick: bool = True, random_objects: bool = False, render_mode: str = None):
        super().__init__()

        self.render_mode = render_mode
        self.width = 10.0
        self.height = 10.0

        self.just_pick = just_pick
        self.random_objects = random_objects

        # Define action and observation spaces
        n_actions = 5 if self.just_pick else 6
        self.action_space = spaces.Discrete(n_actions)
        self.num_shelves = 3
        
        # NUEVO: Espacio de observación aumentado de 11 a 15 elementos
        self.observation_space = spaces.Box(
            low=-10,  # Permitir valores negativos para direcciones
            high=10,
            shape=(15 + 4 * self.num_shelves,),  # De 11 a 15 elementos
            dtype=np.float32
        )

        # Warehouse layout
        self.shelves = [(1.9, 1.0, 0.2, 5.0), (4.9, 1.0, 0.2, 5.0), (7.9, 1.0, 0.2, 5.0)]
        self.delivery_area = (2.5, 9, 5.0, 1.0)  # Corregido: altura 1.0 según el PDF
        self.delivery_area_center = (self.delivery_area[0] + self.delivery_area[2]/2, 
                                     self.delivery_area[1] + self.delivery_area[3]/2)

        # Variables used to calculate reward depending on distance to objetive
        self.closest_obj_idx = None
        self.last_distance = None
        
        # Agent properties
        self.agent_radius = 0.2
        self.agent_velocity = 0.5
        self.pickup_distance = 0.6

        # **NUEVAS VARIABLES MEJORADAS PARA DETECTAR BUCLES**
        self.action_history = []  # Historial de acciones
        self.history_size = 10   # Tamaño del historial
        self.position_history = []  # Historial de posiciones
        self.visited_positions = []  # NUEVO: posiciones visitadas redondeadas
        self.position_history_size = 20  # NUEVO: tamaño del historial de posiciones
        self.stuck_counter = 0  # Contador de veces atascado
        
        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = self._get_random_empty_position()

        if self.random_objects:
            self.object_positions = [self._get_random_position_on_shelf(s) for s in self.shelves]
        else:
            self.object_positions = [(2, 3.0), (5, 2.0), (8, 4.0)]  # Posiciones más accesibles

        # Calculate distances to nearest object
        available_objects = [obj for obj in self.object_positions if obj is not None]
        if available_objects:
            dists = [self._distance(self.agent_pos, obj) for obj in available_objects]
            self.closest_obj_idx = int(np.argmin(dists))
            self.last_distance = dists[self.closest_obj_idx]
        else:
            self.closest_obj_idx = 0
            self.last_distance = 0

        self.agent_has_object = False
        self.delivery = False
        self.collision = False

        # Reset de variables anti-bucle
        self.action_history = []
        self.position_history = []
        self.visited_positions = []  # NUEVO: reset de posiciones visitadas
        self.stuck_counter = 0

        self.steps = 0
        self.max_steps = 200  # Aumentado para dar más tiempo para entrega

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # --- NUEVO SISTEMA DE RECOMPENSAS MEJORADO ---
        STEP_PENALTY = -0.05           # Reducida penalización por paso
        PICK_OBJECT_REWARD = 30.0      # Aumentada recompensa por recoger
        DELIVER_OBJECT_REWARD = 100.0  # MÁXIMA recompensa por entregar
        COLLISION_PENALTY = -10.0      # Penalización por colisión
        PROGRESS_REWARD = 2.0          # Recompensa por acercarse al objetivo
        USELESS_ACTION_PENALTY = -1.5  # Penalización reducida por acciones inútiles
        
        # --- NUEVAS RECOMPENSAS INTERMEDIAS ---
        APPROACH_OBJECT_REWARD = 5.0   # Recompensa por acercarse a objeto
        APPROACH_DELIVERY_REWARD = 3.0 # Recompensa por acercarse a zona de entrega
        LOOP_PENALTY = -8.0           # Penalización por bucles
        STUCK_PENALTY = -4.0          # Penalización por estar atascado

        # --- Initialize state variables ---
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Guardar acción en historial
        self.action_history.append(action)
        if len(self.action_history) > self.history_size:
            self.action_history.pop(0)

        # Target of the agent
        available_objects = [obj for obj in self.object_positions if obj is not None]
        if not self.agent_has_object and available_objects:
            # Encontrar el objeto más cercano disponible
            dists = [self._distance(self.agent_pos, obj) for obj in available_objects]
            self.closest_obj_idx = int(np.argmin(dists))
            target = available_objects[self.closest_obj_idx]
            old_min_obj_dist = min(dists) if dists else 10.0
        else:
            target = self.delivery_area_center
            old_min_obj_dist = 0.0

        old_distance = self._distance(self.agent_pos, target)
        old_delivery_dist = self._distance(self.agent_pos, self.delivery_area_center) if self.agent_has_object else 0.0
        old_pos = self.agent_pos

        # --- Action handling ---
        if action < 4:  # Movement
            new_pos = self._get_new_position(action)
            if not self._is_collision(new_pos):
                self.agent_pos = new_pos
                reward = STEP_PENALTY
                
                # NUEVO: Detectar posiciones visitadas
                current_pos_rounded = (round(self.agent_pos[0], 1), round(self.agent_pos[1], 1))
                self.visited_positions.append(current_pos_rounded)
                if len(self.visited_positions) > self.position_history_size:
                    self.visited_positions.pop(0)
                    
            else:
                self.collision = True
                terminated = True
                reward = COLLISION_PENALTY

        elif action == 4:  # Pick
            if not self.agent_has_object:
                picked = False
                for i, obj_pos in enumerate(self.object_positions):
                    if obj_pos is not None and self._distance(self.agent_pos, obj_pos) <= self.pickup_distance + self.agent_radius:
                        self.agent_has_object = True
                        self.object_positions[i] = None
                        reward = PICK_OBJECT_REWARD
                        picked = True
                        if self.just_pick:
                            terminated = True
                        break
                if not picked:
                    reward = USELESS_ACTION_PENALTY
            else:
                reward = USELESS_ACTION_PENALTY

        elif action == 5:  # Drop
            if self.agent_has_object:
                if self._is_in_area(self.agent_pos, self.delivery_area):
                    reward = DELIVER_OBJECT_REWARD
                    self.delivery = True
                    terminated = True
                else:
                    reward = USELESS_ACTION_PENALTY
                    # NO soltar el objeto fuera de la zona de entrega - mantenerlo
                # self.agent_has_object = False  # Solo se suelta si está en zona de entrega
                # terminated = True  # Solo terminar si entrega exitosa
            else:
                reward = USELESS_ACTION_PENALTY

        # **NUEVO: RECOMPENSAS POR PROGRESO MEJORADAS**
        if not self.agent_has_object and available_objects:
            # Recompensa por acercarse a cualquier objeto disponible
            current_min_obj_dist = min([self._distance(self.agent_pos, obj) for obj in available_objects])
            if current_min_obj_dist < old_min_obj_dist:
                reward += APPROACH_OBJECT_REWARD * (old_min_obj_dist - current_min_obj_dist)
        
        elif self.agent_has_object:
            # Recompensa por acercarse al área de entrega
            current_delivery_dist = self._distance(self.agent_pos, self.delivery_area_center)
            if current_delivery_dist < old_delivery_dist:
                reward += APPROACH_DELIVERY_REWARD * (old_delivery_dist - current_delivery_dist)

        # Recompensa adicional por progreso hacia el objetivo actual
        new_dist = self._distance(self.agent_pos, target)
        reward += (old_distance - new_dist) * PROGRESS_REWARD
        self.last_distance = new_dist

        # **DETECCIÓN DE BUCLES DE MOVIMIENTO MEJORADA**
        if action < 4 and len(self.action_history) >= 4:
            last_4 = self.action_history[-4:]
            
            # Detectar patrones oscilatorios
            if (last_4 == [0, 1, 0, 1] or last_4 == [1, 0, 1, 0] or  # Up-Down
                last_4 == [2, 3, 2, 3] or last_4 == [3, 2, 3, 2]):   # Left-Right
                reward += LOOP_PENALTY
                self.stuck_counter += 1
                if self.stuck_counter >= 4:  # Más tolerante
                    terminated = True
                    reward += LOOP_PENALTY * 1.5

        # **DETECCIÓN DE BUCLES PICK-DROP MEJORADA**
        if not self.just_pick and len(self.action_history) >= 4:
            last_4 = self.action_history[-4:]
            if last_4 == [4, 5, 4, 5] or last_4 == [5, 4, 5, 4]:
                reward += LOOP_PENALTY * 2
                self.stuck_counter += 2
                terminated = True

        # **NUEVA DETECCIÓN DE ESTAR ATASCADO POR POSICIONES VISITADAS**
        if len(self.visited_positions) >= 8:
            recent_positions = self.visited_positions[-8:]
            unique_positions = set(recent_positions)
            if len(unique_positions) <= 3:  # Muy poca variedad de posiciones
                reward += STUCK_PENALTY
                self.stuck_counter += 1
                
                if self.stuck_counter >= 6:  # Más tolerante
                    terminated = True
                    reward += STUCK_PENALTY * 2

        # **PENALIZACIÓN EXPONENCIAL si está atascado muchas veces**
        if self.stuck_counter > 2:
            reward += STUCK_PENALTY * (self.stuck_counter - 2) * 0.3

        if self.steps >= self.max_steps:
            truncated = True
            # Pequeña penalización por timeout
            reward -= 2.0

        # --- Prepare outputs ---
        obs = self._get_obs()
        info = {
            'stuck_counter': self.stuck_counter,
            'action_history': self.action_history.copy(),
            'success': (self.delivery and terminated)
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # NUEVO: Observación aumentada de 11 a 15 elementos
        obs = np.zeros(15 + 4 * self.num_shelves, dtype=np.float32)

        # --- Agent position ---
        obs[0:2] = self.agent_pos

        # --- Object positions ---
        available_objects = [obj for obj in self.object_positions if obj is not None]
        for i in range(3):  # Siempre 3 slots para objetos
            if i < len(available_objects):
                obs[2 + 2*i : 4 + 2*i] = available_objects[i]
            else:
                obs[2 + 2*i : 4 + 2*i] = [-1, -1]  # -1 indica objeto no disponible

        # --- Status flags ---
        obs[8] = float(self.agent_has_object)
        obs[9] = float(self.collision)
        obs[10] = float(self.delivery)

        # --- NUEVAS OBSERVACIONES CRÍTICAS ---
        # Distancia al objeto más cercano (si no tiene objeto)
        available_objects = [obj for obj in self.object_positions if obj is not None]
        if not self.agent_has_object and available_objects:
            dists = [self._distance(self.agent_pos, obj) for obj in available_objects]
            obs[11] = min(dists) if dists else 10.0
        else:
            obs[11] = 0.0
        
        # Distancia al área de entrega (si tiene objeto)
        obs[12] = self._distance(self.agent_pos, self.delivery_area_center) if self.agent_has_object else 0.0
        
        # Dirección al objetivo actual (normalizada)
        if not self.agent_has_object and available_objects:
            # Encontrar el objeto más cercano
            dists = [self._distance(self.agent_pos, obj) for obj in available_objects]
            closest_idx = np.argmin(dists)
            target = available_objects[closest_idx]
        elif self.agent_has_object:
            target = self.delivery_area_center
        else:
            target = self.delivery_area_center  # Fallback
        
        dx, dy = target[0] - self.agent_pos[0], target[1] - self.agent_pos[1]
        dist_to_target = max(self._distance(self.agent_pos, target), 0.1)  # Evitar división por cero
        obs[13] = dx / dist_to_target  # Dirección X normalizada
        obs[14] = dy / dist_to_target  # Dirección Y normalizada

        # --- Shelves info ---
        offset = 15  # Actualizado por el nuevo tamaño
        for i, shelf in enumerate(self.shelves):
            xc = shelf[0] + shelf[2] / 2
            yc = shelf[1] + shelf[3] / 2
            w  = shelf[2]
            h  = shelf[3]

            obs[offset + 4*i + 0] = xc
            obs[offset + 4*i + 1] = yc
            obs[offset + 4*i + 2] = w
            obs[offset + 4*i + 3] = h

        return obs

    def _get_new_position(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            y = min(self.height - self.agent_radius, y + self.agent_velocity)
        elif action == 1:  # Down
            y = max(self.agent_radius, y - self.agent_velocity)
        elif action == 2:  # Left
            x = max(self.agent_radius, x - self.agent_velocity)
        elif action == 3:  # Right
            x = min(self.width - self.agent_radius, x + self.agent_velocity)
        return (x, y)

    def _is_collision(self, pos):
        if (
            pos[0] <= self.agent_radius or
            pos[0] >= self.width - self.agent_radius or
            pos[1] <= self.agent_radius or
            pos[1] >= self.height - self.agent_radius
        ):
            return True

        for shelf in self.shelves:
            if self._is_in_area(pos, shelf, self.agent_radius):
                return True

        return False

    def _get_random_empty_position(self):
        while True:
            pos = (
                np.random.uniform(self.agent_radius, self.width - self.agent_radius),
                np.random.uniform(self.agent_radius, self.height - self.agent_radius),
            )
            if not self._is_collision(pos):
                return pos

    def _get_random_position_on_shelf(self, shelf):
        aux = np.random.uniform(0, 1)
        x = shelf[0] + (0.25 if aux < 0.5 else 0.75) * shelf[2]
        y = np.random.uniform(shelf[1] + 0.5, shelf[1] + shelf[3] - 0.5)
        return (x, y)

    @staticmethod
    def _distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _is_in_area(pos, area, margin=0):
        return (
            area[0] - margin <= pos[0] <= area[0] + area[2] + margin and
            area[1] - margin <= pos[1] <= area[1] + area[3] + margin
        )

    def render(self):
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")

        # Shelves
        for s in self.shelves:
            self.ax.add_patch(Rectangle(s[:2], s[2], s[3], fill=False, edgecolor="brown"))

        # Delivery area
        self.ax.add_patch(Rectangle(self.delivery_area[:2], self.delivery_area[2], self.delivery_area[3],
                                    fill=True, facecolor="lightgreen", edgecolor="green", alpha=0.5))

        # Objects
        for obj in self.object_positions:
            if obj is not None:
                self.ax.add_patch(Circle(obj, radius=0.2, color="blue"))

        # Agent
        color = "red" if self.agent_has_object else "orange"
        self.ax.add_patch(Circle(self.agent_pos, radius=self.agent_radius, color=color))

        # Mostrar contador de atascamiento
        plt.title(f"WarehouseEnv - Stuck: {self.stuck_counter}")
        plt.draw()
        plt.pause(0.05)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None


if __name__ == "__main__":
    env = WarehouseEnv(just_pick=True, random_objects=False, render_mode="human")
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Stuck: {info.get('stuck_counter', 0)}")
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()