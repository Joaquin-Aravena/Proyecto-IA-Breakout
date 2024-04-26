import gymnasium as gym
import keyboard

env = gym.make('ALE/Breakout-v5', render_mode="human")

observation, info = env.reset()

while True:
    if keyboard.is_pressed('a'):
        action = 3
    elif keyboard.is_pressed('d'):
        action = 2
    elif keyboard.is_pressed('w'):
        action = 1
    else:
        action = 0

    observation, reward, terminated, truncated, info = env.step(action)

    # reward: puntaje obtenido en el step (aumenta el frame en que se rompe un bloque)
    # terminated: devuelve true cuando se acaba el juego (5 vidas perdidas)
    # truncated: devuelve true cuando se cumple condición de truncamiento (típicamente límite de tiempo o que agente no se salga de bounds. Definido por ambiente)
    # info: arreglo de 3 elementos que contiene {numero_vidas, frame_episodio, frame}

    if terminated or truncated:
        observation, info = env.reset()

env.close()
