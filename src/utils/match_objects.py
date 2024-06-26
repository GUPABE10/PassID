class Player:
    """
        team: es el id del cluster
    """
    def __init__(self, player_id: int, team: str):
        self.id = player_id # Es el mismo que tracked Id
        self.team = team

    def __str__(self):
        return f"Player(player_id={self.id}, team={self.team})"





# Ejemplo de uso
# tracked_objects = [...]  # Lista de objetos rastreados
# image = ...  # Imagen cargada con OpenCV
# balls = check_ball_possession(tracked_objects, image)
# for ball in balls:
#     print("Balón en posesión:", ball.in_possession)



class Ball:
    def __init__(self, id):
        self.id = id
        self.inPossession = False
        self.framesInPossession = 0

    def __str__(self):
        return f"Ball(id={self.id})"