# take5.py
# OpenSpiel implementation of Wolfgang Kramer’s “Take 5!” (a.k.a. 6 nimmt!)
#
# Rules source: AMIGO rule book, item #18415 V1‑0818, pages 2‑6:contentReference[oaicite:8]{index=8}.

from typing import Dict, List, Tuple
import random
import numpy as np          # Only used for observation tensors
import pyspiel


# ----------  Helper constants -------------------------------------------------

NUM_CARDS = 104                       # 1 … 104
NUM_PLAYERS = 4
MAX_ROW_LEN = 5                       # a 6th card triggers “Take 5”
CARDS_PER_PLAYER = 10                 # one round only
ROWS = 4

GAME_TYPE = pyspiel.GameType(
    short_name="take5",
    long_name="Take 5 (6 nimmt!)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,  # we emulate the simultaneous reveal inside the state
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,   # each player has independent score
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_tensor=True,
    provides_information_state_string=True,
    provides_observation_tensor=True,
    provides_observation_string=True,
)
GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_CARDS + ROWS,  # 104 cards (0‑103) + 4 “choose‑row” actions
    max_chance_outcomes=0,
    num_players=NUM_PLAYERS,
    min_utility=-66,                       # hard bound from rule book
    max_utility=0,
    utility_sum=0.0,
    max_game_length=CARDS_PER_PLAYER + CARDS_PER_PLAYER * NUM_PLAYERS * 2,  # loose bound
)

# tic-tac-toe example
# _GAME_TYPE = pyspiel.GameType(
#     short_name="python_tic_tac_toe",
#     long_name="Python Tic-Tac-Toe",
#     dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
#     chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
#     information=pyspiel.GameType.Information.PERFECT_INFORMATION,
#     utility=pyspiel.GameType.Utility.ZERO_SUM,
#     reward_model=pyspiel.GameType.RewardModel.TERMINAL,
#     max_num_players=NUM_PLAYERS,
#     min_num_players=NUM_PLAYERS,
#     provides_information_state_string=True,
#     provides_information_state_tensor=False,
#     provides_observation_string=True,
#     provides_observation_tensor=True,
#     parameter_specification={})
# _GAME_INFO = pyspiel.GameInfo(
#     num_distinct_actions=_NUM_CELLS,
#     max_chance_outcomes=0,
#     num_players=2,
#     min_utility=-1.0,
#     max_utility=1.0,
#     utility_sum=0.0,
#     max_game_length=_NUM_CELLS)

def bullheads(card: int) -> int:
    """Return the penalty (“bullhead”) value printed on a card."""
    if card == 55:              # special – both 5 × 11 and printed as 7 heads
        return 7
    if card % 11 == 0:          # multiple of 11
        return 5
    if card % 10 == 0:          # multiple of 10 (but not 55)
        return 3
    if card % 5 == 0:           # multiple of 5 (but not 55 / 10‑multiple)
        return 2
    return 1                    # all others


# ----------  OpenSpiel game & state classes -----------------------------------

class TakeFiveGame(pyspiel.Game):
    """Game object shared by all states."""

    def __init__(self, params: Dict[str, str] | None = None):
        params = params or {}
        # self._num_players = int(params.get("players", 4))   # 2-10 in the box rules
        self._num_players = NUM_PLAYERS
        super().__init__(GAME_TYPE, GAME_INFO, params or dict())

    # Factory:
    def new_initial_state(self) -> 'TakeFiveState':
        return TakeFiveState(self, self._num_players)

    # Pretty‑print:
    def make_py_observer(self, iig_obs_type=None, params=None):
        return TakeFiveObserver()


class TakeFiveState(pyspiel.State):
    """A single round of Take 5."""

    def __init__(self, game: TakeFiveGame, num_players: int):
        super().__init__(game)
        self._num_players = num_players
        self._rng = random.Random()                         # OpenSpiel sets seed globally
        self._deck = list(range(1, NUM_CARDS + 1))
        self._rng.shuffle(self._deck)

        # Hands
        self._hands: List[List[int]] = [
            sorted(self._deck[i * CARDS_PER_PLAYER:(i + 1) * CARDS_PER_PLAYER])
            for i in range(num_players)
        ]
        self._deck_ptr = num_players * CARDS_PER_PLAYER    # remaining deck unused this round

        # Four initial rows – next four cards face‑up (rule p. 2):contentReference[oaicite:9]{index=9}
        self._rows: List[List[int]] = [[self._draw()] for _ in range(ROWS)]

        # Penalty piles per player
        self._bull_piles: List[List[int]] = [[] for _ in range(num_players)]

        # Simultaneous selection buffer
        self._chosen: Dict[int, int] = {}      # player → card
        self._pending_sequence: List[Tuple[int, int]] = []  # (card, player) sorted ascending

        # Phase control
        self._phase = "select"                 # "select", "resolve", "choose_row", "terminal"
        self._current_player = 0               # only used in choose_row phase
        self._row_choice_needed_for: int = -1  # card being placed when choose_row active

    # ------------------------------------------------------------------ helpers
    def _draw(self) -> int:
        card = self._deck[self._deck_ptr]
        self._deck_ptr += 1
        return card

    def _row_penalty(self, row: List[int]) -> int:
        return sum(bullheads(c) for c in row)

    def _closest_row(self, card: int) -> int:
        """Return the row index that card must go to, or -1 if card < min(row ends)."""
        candidates = [(card - r[-1], idx) for idx, r in enumerate(self._rows) if card > r[-1]]
        if not candidates:
            return -1
        # least positive difference
        return min(candidates)[1]

    # --------------------------------------------------------- OpenSpiel core
    def current_player(self) -> int:
        if self._phase == "select":
            return pyspiel.PlayerId.SIMULTANEOUS  # each chooses a card
        elif self._phase == "choose_row":
            return self._current_player
        elif self._phase == "terminal":
            return pyspiel.PlayerId.TERMINAL
        else:  # resolving – environment acting
            return pyspiel.PlayerId.CHANCE  # treated as chance to keep MCTS happy

    def legal_actions(self, player: int) -> List[int]:
        if self._phase == "select":
            # card ids are 0–103; action is card‑1
            return [c - 1 for c in self._hands[player]]
        elif self._phase == "choose_row" and player == self._current_player:
            # Actions 104…107 map to row 0…3
            return list(range(NUM_CARDS, NUM_CARDS + ROWS))
        return []

    def apply_actions(self, joint_actions: List[int]):
        """Only called during simultaneous select phase."""
        assert self._phase == "select"
        for p, act in enumerate(joint_actions):
            if act == pyspiel.INVALID_ACTION:
                raise ValueError("All players must submit an action")
            card = act + 1
            self._hands[p].remove(card)
            self._chosen[p] = card

        # create processing queue
        self._pending_sequence = sorted((c, p) for p, c in self._chosen.items())
        self._chosen.clear()
        self._phase = "resolve"
        self._process_next_in_queue()

    def apply_action(self, action: int):
        """Sequential actions only used for choose_row decisions."""
        if self._phase != "choose_row":
            raise ValueError("Unexpected sequential action")
        row_idx = action - NUM_CARDS
        assert 0 <= row_idx < ROWS
        self._take_row(self._current_player, row_idx, replacement_card=self._row_choice_needed_for)
        self._row_choice_needed_for = -1
        self._phase = "resolve"
        self._process_next_in_queue()

    # ---------------------------------------------------------------- private
    def _take_row(self, player: int, row_idx: int, replacement_card: int):
        """Player takes row, collects penalty, and starts new row with replacement_card."""
        self._bull_piles[player].extend(self._rows[row_idx])
        self._rows[row_idx] = [replacement_card]

    def _place_card(self, player: int, card: int):
        row_idx = self._closest_row(card)
        if row_idx == -1:
            # player chooses which row to take – defer to choose_row phase
            self._phase = "choose_row"
            self._current_player = player
            self._row_choice_needed_for = card
            return
        # Would placing make 6th card?
        if len(self._rows[row_idx]) == MAX_ROW_LEN:
            self._take_row(player, row_idx, replacement_card=card)
        else:
            self._rows[row_idx].append(card)

    def _process_next_in_queue(self):
        while self._phase == "resolve" and self._pending_sequence:
            card, player = self._pending_sequence.pop(0)
            self._place_card(player, card)
            if self._phase == "choose_row":
                return  # wait for player input
        # Finished all revealed cards
        if all(len(h) == 0 for h in self._hands):
            self._phase = "terminal"
        else:
            self._phase = "select"      # next simultaneous reveal

    # ----------------------------------------------------------- RL interface
    def _collect_bullheads(self) -> List[int]:
        return [sum(bullheads(c) for c in pile) for pile in self._bull_piles]

    def rewards(self) -> List[float]:
        # sparse – only at terminal
        return [0.0] * self._num_players

    def returns(self) -> List[float]:
        if not self.is_terminal():
            return [0.0] * self._num_players
        # We *maximize* utility in OpenSpiel, so return negative penalty
        return [-p for p in self._collect_bullheads()]

    def is_terminal(self) -> bool:
        return self._phase == "terminal"

    # ------------- Observation / information‑state (minimal but functional) --
    def _hand_tensor(self, player: int) -> np.ndarray:
        vec = np.zeros(NUM_CARDS, dtype=np.float32)
        for c in self._hands[player]:
            vec[c - 1] = 1.0
        return vec

    def _rows_tensor(self) -> np.ndarray:
        out = np.zeros((ROWS, MAX_ROW_LEN), dtype=np.float32)
        for r, row in enumerate(self._rows):
            for idx, card in enumerate(row):
                out[r, idx] = card / NUM_CARDS   # scale roughly to [0,1]
        return out.flatten()

    def observation_tensor(self, player: int) -> np.ndarray:
        return np.concatenate([self._hand_tensor(player),
                               self._rows_tensor()])

    def observation_string(self, player: int) -> str:
        s = f"Rows: {self._rows}\nHand: {self._hands[player]}\nBull pile: {self._bull_piles[player]}"
        return s

    # -------------------------------------------------------------- debugging
    def __str__(self):
        rep = ["TABLE:"]
        for r in self._rows:
            rep.append(" ".join(f"{c:>2}" for c in r))
        for p in range(self._num_players):
            rep.append(f"P{p} hand {self._hands[p]}")
            rep.append(f"P{p} pile {self._bull_piles[p]}")
        return "\n".join(rep)


class TakeFiveObserver(pyspiel.Observer):
    """String observer for human debugging – not used by RL."""
    def __init__(self):
        super().__init__(iig_obs_type=None)

    def set_from(self, state: TakeFiveState, player: int):
        self._string = state.observation_string(player)

    def string_from(self, state: TakeFiveState, player: int) -> str:
        return state.observation_string(player)
