import gymnasium as gym
import keyboard

env = gym.make('ALE/Breakout-v5', render_mode="human")

observation, info = env.reset()

# env.action_space.n = numero de acciones
# 4

# env.unwrapped.get_action_meanings() = arreglo  de acciones (con sus nombres)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

while True:
    if keyboard.is_pressed('a'):    # izquierda
        action = 3
    elif keyboard.is_pressed('d'):  # derecha
        action = 2
    elif keyboard.is_pressed('w'):  # iniciar juego
        action = 1
    else:
        action = 0

    observation, reward, terminated, truncated, info = env.step(action)

    # observation: en modo rgb (modo default de observación) es matriz de 210x160x3
    # Dimensiones de pantalla es 210 ancho, 160 alto (cada par es un pixel), el 3 proviene de los tres canales r, g y b.
    # reward: puntaje obtenido en el step (devuelve != 0 solo en el instante en que se rompe un bloque)
    # terminated: devuelve true cuando se acaba el juego (5 vidas perdidas)
    # truncated: devuelve true cuando se cumple condición de truncamiento (típicamente límite de tiempo o que agente no se salga de bounds. Definido por ambiente)
    # info: arreglo de 3 elementos que contiene {numero_vidas, frame_episodio, frame}

    if terminated or truncated:
        observation, info = env.reset()

env.close()
