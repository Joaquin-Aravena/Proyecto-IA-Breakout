# Atari Breakout

## Proyecto Semestral de Inteligencia Artificial
Este proyecto semestral para el ramo de Inteligencia Artificial de la Universidad consiste en la implementación de un agente inteligente para el juego Breakout utilizando el entorno de Gym.
##Descripción
Breakout es...

### Ambiente
- **Tipo:** Observacional, Determinista, Estático, Discreto, Secuencial
- **Agente:** Singular

## Representación de las acciones:

En nuestro juego, las acciones se representan mediante números discretos que indican la dirección del movimiento de la paleta. Es simple y directo: 

- **0**: No mover la paleta.
- **1**: Mover la paleta hacia la izquierda.
- **2**: Mover la paleta hacia la derecha.

Aquí tienes una vista rápida de cómo se relacionan los números con las acciones:

```python
acciones = {
    0: "No mover",
    1: "Mover izquierda",
    2: "Mover derecha"
}
