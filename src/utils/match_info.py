import cv2

from utils.match_objects import Player, Ball

class VideoInfo:
    def __init__(self, video_path, frame_rate: int = 5):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.width, self.height, self.duration = self.get_video_info()

    def get_video_info(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        frame_rate_original = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / frame_rate_original
        
        cap.release()
        return width, height, duration

    def __str__(self):
        return f"VideoInfo(frame_rate={self.frame_rate}, width={self.width}, height={self.height}, duration={self.duration})"


class Match:
    def __init__(self):
        self.players = {}
        self.extras = set() # Evita Duplicidad
        self.ball = None

    def add_player(self, player_id, team):
        self.players[player_id] = Player(player_id, team)
        
    def add_extra_people(self, extra_id):
        self.extras.add(extra_id)
        
    def assignBall(self, ball_id):
        self.ball = Ball(ball_id)
    
        
    def assign_ball_to_match(self, tracked_objects):
        ball_ids = [obj.id for obj in tracked_objects if obj.label == 2]
        
        if self.ball is None:
            # Caso cuando no hay ninguna pelota asignada previamente
            if len(ball_ids) == 1:
                self.assignBall(ball_id=ball_ids[0])
            elif len(ball_ids) > 1:
                self.assignBall(ball_id=0)
        else:
            # Caso cuando ya hay una pelota asignada
            current_ball_id = self.ball.id
            if len(ball_ids) == 1:
                if ball_ids[0] != current_ball_id:
                    self.assignBall(ball_id=ball_ids[0])
            elif len(ball_ids) > 1:
                # Si hay más de una pelota detectada, se mantiene la pelota actual
                pass
            elif len(ball_ids) == 0:
                # Si no se detecta ninguna pelota, se podría decidir si se debe hacer algo
                pass

    # def __str__(self):
    #     players_str = "\n".join(str(player) for player in self.players.values())
    #     # ball_str = str(self.ball)
    #     return f"Match(\nPlayers:\n{players_str})"
    
    def __str__(self):
        players_str = ', '.join(
            f"Player(id={player.id}, team={player.team})" 
            for player in self.players.values()
        )
        extras_str = ', '.join(str(extra) for extra in self.extras)
        ball_str = f"Ball(id={self.ball.id}, inPossession={self.ball.inPossession})" if self.ball else "No ball"
        
        return (f"Match(\n"
                f"  Players: [{players_str}],\n"
                f"  Extras Ids: [{extras_str}],\n"
                f"  Ball: {ball_str}\n"
                f")")



    
    
"""
from src.utils.match_info import VideoInfo, Match

# Información del video
video_info = VideoInfo(frame_rate=30, width=1920, height=1080, duration=3600)
print(video_info)

# Información del partido
match = Match()
match.add_player(player_id=1, team='A', initial_position=(100, 50))
match.add_player(player_id=2, team='B', initial_position=(150, 60))
match.set_ball_position((120, 55))

# Actualizar posiciones
match.update_player_position(player_id=1, new_position=(110, 55))
match.set_ball_position((125, 58))

print(match)


"""