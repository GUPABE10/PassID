import cv2
import numpy as np

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
        self.lastPlayerWithBall = None

    def add_player(self, player_id, team):
        self.players[player_id] = Player(player_id, team)
        
    def add_extra_people(self, extra_id):
        self.extras.add(extra_id)
        
    def assignBall(self, ball_id):
        self.ball = Ball(ball_id)
    
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
    

    def is_green_background(image, bbox, padding=10):
        # Expandir el bounding box
        x1, y1, x2, y2 = bbox
        height, width, _ = image.shape

        x1_exp = max(0, x1 - padding)
        y1_exp = max(0, y1 - padding)
        x2_exp = min(width, x2 + padding)
        y2_exp = min(height, y2 + padding)
        
        # Crear una máscara para el bounding box expandido
        expanded_box = np.zeros_like(image[y1_exp:y2_exp, x1_exp:x2_exp])
        cv2.rectangle(expanded_box, (x1 - x1_exp, y1 - y1_exp), (x2 - x1_exp, y2 - y1_exp), (255, 255, 255), thickness=-1)
        
        # Extraer la región expandida de la imagen original
        expanded_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        # Restar el bounding box original
        mask = cv2.inRange(expanded_box, (255, 255, 255), (255, 255, 255))
        surrounding_region = cv2.bitwise_and(expanded_region, expanded_region, mask=mask)

        # Convertir a espacio de color HSV
        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color verde en HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Crear máscara para el color verde
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calcular la proporción de píxeles verdes
        green_ratio = np.sum(green_mask) / (green_mask.size * 255)
        
        # Determinar si es suficiente verde
        return green_ratio > 0.5


    def assign_ball_to_match(self, tracked_objects, image, padding=10):
        ball_candidates = []
        for obj in tracked_objects:
            if obj.label == 2:
                bbox = obj.estimate.flatten().tolist()
                if self.is_green_background(image, bbox, padding):
                    ball_candidates.append(obj.id)
        
        if self.ball is None:
            if len(ball_candidates) == 1:
                self.assignBall(ball_id=ball_candidates[0])
            elif len(ball_candidates) > 1:
                self.assignBall(ball_id=0)
        else:
            current_ball_id = self.ball.id
            if len(ball_candidates) == 1:
                if ball_candidates[0] != current_ball_id:
                    self.assignBall(ball_id=ball_candidates[0])
            elif len(ball_candidates) > 1:
                pass

    

    
    def update_ball_possession(self, tracked_objects, video_info):
        if self.ball is None:
            # print(f"Balón no identificado aún: {self.ball}")
            return
        elif self.ball.id == 0:
            # print(f"Balón no identificado aún: {self.ball}")
            return
        
        # Posisicion del balon
        ball_position = None
        for obj in tracked_objects:
            if obj.id == self.ball.id:
                ball_position = obj
                break
        
        if ball_position is None:
            # print(f"No se encontró al balón dentro de Tracked_objects. Id = {self.ball.id}")
            # print(f"Actual Tracked Objects: {tracked_objects}")
            return
        
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_position.estimate.flatten().tolist()
        ball_center_x = (ball_x1 + ball_x2) / 2
        ball_center_y = (ball_y1 + ball_y2) / 2
        
        # Encotnrar jugador mas cercano
        closest_player = None
        min_distance = float('inf')
        
        player_position = None
        for player_id, player in self.players.items():
            # 1. Get position of player_id
            for obj in tracked_objects:
                if obj.id == player_id:
                    player_position = obj
                    break

            # ! A veces player_position no se encuentra
            if player_position is not None:
                player_x1, player_y1, player_x2, player_y2 = player_position.estimate.flatten().tolist()
            

                # Si balon izquierda
                # Determinar el punto más cercano del jugador al balón
                if ball_center_x < player_x1:
                    player_x, player_y = player_x1, player_y2
                else:
                    player_x, player_y = player_x2, player_y2
                
                # Euclidian Distance
                distance = ((ball_center_x - player_x) ** 2 + (ball_center_y - player_y) ** 2) ** 0.5
                # Distance / Diagonal de video
                distance_proportional = distance / ((video_info.width ** 2 + video_info.height ** 2) ** 0.5)
                
                if distance_proportional < min_distance:
                    min_distance = distance_proportional
                    closest_player = player
        
        threshold = 0.03  # Experimental puede ser menor a 0.05 pero no mayor.
        
        # Si hay un jugador cercano
        if min_distance < threshold:

            # Si no hay jugador con pelota
            if self.lastPlayerWithBall is None:
                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1

            # Si el mismo jugador sigue teniendo la pelota
            elif closest_player.id == self.lastPlayerWithBall.id:
                self.ball.framesInPossession+=1

            # Si es diferente jugador
            else:
                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1


            if self.ball.framesInPossession >= 3:
                self.ball.inPossession = True

        else:
            self.ball.inPossession = False
            self.playerWithBall = None



    
    
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