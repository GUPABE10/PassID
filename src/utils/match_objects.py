class Player:
    """
        team: puede ser el color dominante del equipo, de esta manera la persona que lo ejecuta puede diferenciar
        }
    """
    def __init__(self, player_id: int, team: str):
        self.player_id = player_id # Es el mismo que tracked Id
        self.team = team

    def __str__(self):
        return f"Player(player_id={self.player_id}, team={self.team})"



# class Ball:
#     def __init__(self, initial_position):
#         self.position = initial_position
#         self.tracking_history = []

#     def __str__(self):
#         return f"Ball(position={self.position})"