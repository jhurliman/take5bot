import pyspiel
import openspiel_take5

pyspiel.register_game(openspiel_take5.GAME_TYPE, openspiel_take5.TakeFiveGame)
g = pyspiel.load_game("take5")
state = g.new_initial_state()
print(state)                                      # human‑readable dump
print(state.observation_string(0))
