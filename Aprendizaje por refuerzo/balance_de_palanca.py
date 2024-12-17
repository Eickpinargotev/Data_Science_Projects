import gym
import numpy as np
import math
import matplotlib.pyplot as plt  # Importa matplotlib para graficar

# Crear el entorno de CartPole
env = gym.make('CartPole-v1', render_mode='rgb_array')  # Sin renderizado durante el entrenamiento

# Parámetros de aprendizaje
alpha = 0.05        # Tasa de aprendizaje ajustada a 0.1
gamma = 1.0        # Factor de descuento ajustado a 1.0
epsilon = 0.2      # Probabilidad de exploración inicial ajustada a 0.2
epsilon_decay = 0.999  # Factor de decaimiento de epsilon ajustado a 0.999
epsilon_min = 0.01      # Mínimo valor de epsilon
episodes = 10000        # Número de episodios aumentado a 15,000

bins = [10, 10, 30, 30]  # Discretización de cada dimensión del estado ajustada a 30 bins

# Discretización del espacio de estados
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-10,10]   # Limitar velocidad del carro ajustado a [-3, 3]
state_bounds[3] = [-10, 10] # Limitar velocidad angular ajustado a [-10, 10]


# Crear los bordes de los bins para cada dimensión
bins_edges = [np.linspace(b[0], b[1], num + 1)[1:-1] for b, num in zip(state_bounds, bins)]

print("Límites de los estados discretizados:")
for i, (low, high) in enumerate(state_bounds):
    print(f"Dimensión {i}: {low} a {high} con {bins[i]} bins")

def discretize_state(state):
    """Convierte un estado continuo en un estado discretizado."""
    return tuple(np.digitize(s, b) for s, b in zip(state, bins_edges))

num_states = tuple(len(b) + 1 for b in bins_edges)  # Número de estados discretizados
num_actions = env.action_space.n

# Inicializar la tabla Q con valores aleatorios entre 0 y 1
Q = np.random.uniform(low=0, high=1, size=num_states + (num_actions,))

scores = []  # Lista para almacenar las puntuaciones de cada episodio

for episode in range(1, episodes + 1):
    state_continuous, _ = env.reset()  # Reinicia el entorno
    state = discretize_state(state_continuous)
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        # Seleccionar acción
        if episode <= 500:
            # Durante los primeros 500 episodios, selección aleatoria para exploración completa
            action = np.random.choice(num_actions)
        elif episode > 7000:
            # Después del episodio 7000, decaer epsilon multiplicativamente
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # Exploración
            else:
                action = np.random.choice(np.where(Q[state] == np.max(Q[state]))[0])  # Explotación con manejo de múltiples acciones óptimas
        else:
            # Entre el episodio 501 y 7000, usar la política ε-greedy con epsilon fijo
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # Exploración
            else:
                action = np.random.choice(np.where(Q[state] == np.max(Q[state]))[0])  # Explotación con manejo de múltiples acciones óptimas

        # Tomar acción
        next_state_continuous, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state_continuous)
        score += reward

        # Actualizar Q
        if not (terminated or truncated):
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
        else:
            # En estado terminal, no considerar recompensas futuras
            td_target = reward
            td_error = td_target - Q[state][action]

        Q[state][action] += alpha * td_error

        # Actualizar estado
        state = next_state

    # Almacenar la puntuación del episodio
    scores.append(score)

    # Mostrar resultados cada 1000 episodios
    if episode % 1000 == 0 or episode <= 10:
        print(f'Episodio: {episode}, Puntuación: {score}, Epsilon: {epsilon:.4f}')

env.close()

# Función para simular la política aprendida
def probar_politica_aprendida(Q, env, bins_edges, num_steps=1000, delay=0.05):
    import time
    env_render = gym.make('CartPole-v1', render_mode='human')
    state_continuous, _ = env_render.reset()
    state = discretize_state(state_continuous)
    obtained_rewards = []
    for _ in range(num_steps):
        action = np.random.choice(np.where(Q[state] == np.max(Q[state]))[0])
        state_continuous, reward, terminated, truncated, _ = env_render.step(action)
        obtained_rewards.append(reward)
        time.sleep(delay)
        if terminated or truncated:
            break
        state = discretize_state(state_continuous)
    env_render.close()

def simulate_learned_policy(Q, env, bins_edges, num_episodes=100, num_steps=1000):
    sum_rewards = []
    for episode in range(num_episodes):
        state_continuous, _ = env.reset()
        state = discretize_state(state_continuous)
        terminated = False
        truncated = False
        total_reward = 0
        for step in range(num_steps):
            # Selección de acción óptima con manejo de múltiples acciones óptimas
            max_actions = np.where(Q[state] == np.max(Q[state]))[0]
            action = np.random.choice(max_actions)
            
            # Tomar acción
            state_continuous, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Actualizar estado
            state = discretize_state(state_continuous)
            
            if terminated or truncated:
                break
        sum_rewards.append(total_reward)
    return sum_rewards

# Función para simular una política aleatoria
def simulate_random_policy(env, num_episodes=100, num_steps=1000):
    sum_rewards = []
    for episode in range(num_episodes):
        state_continuous, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            state_continuous, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        sum_rewards.append(total_reward)
    return sum_rewards

# Simular la política aprendida
print("\nSimulando la política aprendida...")
probar_politica_aprendida(Q, env, bins_edges)
obtained_rewards_optimal = simulate_learned_policy(Q, env, bins_edges)

# Simular la política aleatoria
print("Simulando la política aleatoria...")
obtained_rewards_random = simulate_random_policy(env)

# Graficar las puntuaciones por episodio
plt.figure(figsize=(12, 6))
plt.plot(scores, label='Puntuación por Episodio')

# Calcular y graficar la media móvil para suavizar la curva
window = 50
if len(scores) >= window:
    rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, episodes), rolling_mean, color='red', label=f'Media Móvil ({window} episodios)')

plt.xlabel('Episodio')
plt.ylabel('Puntuación')
plt.title('Progreso del Aprendizaje del Modelo en CartPole')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el histograma de recompensas de la política aleatoria
plt.figure(figsize=(12, 6))
plt.hist(obtained_rewards_random, bins=30, color='gray', edgecolor='black')
plt.xlabel('Suma de Recompensas')
plt.ylabel('Frecuencia')
plt.title('Histograma de Sumas de Recompensas - Política Aleatoria')
plt.grid(True)
plt.show()

# Graficar el histograma de recompensas de la política aprendida
plt.figure(figsize=(12, 6))
plt.hist(obtained_rewards_optimal, bins=30, color='gray', edgecolor='black')
plt.xlabel('Suma de Recompensas')
plt.ylabel('Frecuencia')
plt.title('Histograma de Sumas de Recompensas - Política Aprendida')
plt.grid(True)
plt.show()
