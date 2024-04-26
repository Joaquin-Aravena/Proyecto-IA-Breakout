# Atari Breakout

## Proyecto Semestral de Inteligencia Artificial

Este proyecto semestral, desarrollado como parte del curso de Inteligencia Artificial en la Universidad de Concepción, consiste en la implementación de un agente inteligente para el juego Breakout utilizando el entorno de Gym.

## Descripción

Breakout es un juego clásico de Atari donde el jugador controla una paleta en la parte inferior de la pantalla y debe rebotar una pelota para destruir una pared de ladrillos en la parte superior. El objetivo es eliminar todos los ladrillos sin dejar que la pelota caiga al vacío. Es un desafío que requiere tanto habilidad como estrategia.

##Requerimientos:
Es necesario tener instalado:
  - Python 3.11.0
### Lanzamiento para desarrollo
1. Clonar el repositorio
```
pip install gymnasium
```
```pip install ale-py
```
```pip install gymnasium[atari]
```
```pip install gymnasium[accept-rom-license]
```

### Ambiente

- **Tipo de Ambiente:**
  - **Observable:** Se puede observar el estado completo del juego en la pantalla.
  - **Determinista:** Las acciones y sus consecuencias son predecibles.
  - **Episódico:** Cada partida comienza desde cero y termina cuando el jugador pierde todas sus vidas o completa el objetivo.
  - **Dinámico:** El estado del juego cambia continuamente en respuesta a las acciones del jugador.
  - **Discreto:** Tanto las acciones como los estados del juego se representan mediante valores discretos.
  - **Agente:** Singular
  
### Representación del estado del juegp
```python
estado_juego = {
    "posicion_paleta": (x, y),  # Coordenadas de la paleta
    "posicion_pelota": (x, y),  # Coordenadas de la pelota
    "disposicion_ladrillos": [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], ...],  # Matriz que representa los ladrillos
    "direccion_pelota": (dx, dy),  # Dirección de la pelota
    "velocidad_pelota": v  # Velocidad de la pelota
}
```

### Representación de las Acciones

En nuestro juego, las acciones se representan mediante números discretos que indican la dirección del movimiento de la paleta. Es simple y directo:

- **0:** No mover la paleta.
- **1:** Mover la paleta hacia la izquierda.
- **2:** Mover la paleta hacia la derecha.

Aquí tienes una vista rápida de cómo se relacionan los números con las acciones:

```python
acciones = {
    0: "No mover",
    1: "Mover izquierda",
    2: "Mover derecha"
}
