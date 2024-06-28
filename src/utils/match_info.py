import cv2
import numpy as np

from utils.match_objects import Player, Ball, Pass

class VideoInfo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.width, self.height, self.duration, self.frame_rate, self.frame_count = self.get_video_info()

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
        return width, height, duration, frame_rate_original, frame_count
    
    def frames_to_seconds(self, frames):
        if frames < 0 or frames > self.frame_count:
            raise ValueError("Frame number out of range")
        return frames / self.frame_rate

    def __str__(self):
        return f"VideoInfo(frame_rate={self.frame_rate}, width={self.width}, height={self.height}, duration={self.duration})"


class Match:
    def __init__(self):
        self.players = {}
        self.extras = set() # Evita Duplicidad
        self.ball = None
        self.lastPlayerWithBall = None
        self.newPass = None
        self.mediumPlayer = None # Para guardar a algun jugador de paso

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
    

    def is_green_background(self, image, bbox, padding=10):
        x1, y1, x2, y2 = bbox
        height, width, _ = image.shape

        x1_exp = max(0, x1 - padding)
        y1_exp = max(0, y1 - padding)
        x2_exp = min(width, x2 + padding)
        y2_exp = min(height, y2 + padding)
        
        expanded_box = np.zeros_like(image[y1_exp:y2_exp, x1_exp:x2_exp])
        cv2.rectangle(expanded_box, (x1 - x1_exp, y1 - y1_exp), (x2 - x1_exp, y2 - y1_exp), 255, thickness=-1)
        
        expanded_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        mask = cv2.inRange(expanded_box, 255, 255)
        surrounding_region = cv2.bitwise_and(expanded_region, expanded_region, mask=mask)

        # Visualizar las regiones
        # self.visualize_bbox(image, bbox, padding)

        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_ratio = np.sum(green_mask) / (green_mask.size * 255)
        
        # print(f"Bounding Box: {bbox}")
        # print(f"Green Ratio: {green_ratio}")

        return green_ratio > 0.2


    def assign_ball_to_match(self, tracked_objects, image, padding=10):
        ball_candidates = []
        for obj in tracked_objects:
            if obj.label == 2:
                bbox = obj.estimate.flatten().tolist()
                # if self.is_green_background(image, bbox, padding):
                #     ball_candidates.append(obj.id)
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

        # print(f"Ball: {self.ball}")

    

    
    def update_ball_possession(self, tracked_objects, video_info, frame_number):
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

            self.ball.inPossession = False
            # self.ball.framesInPossession = 0
            self.ball.initFrameNumber = frame_number
            self.ball.framesInTransit += 1  # Para saber cuanto tiempo tardó el pase
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
        
        # print(f"Closest Player: {closest_player}")
        # print(f"Last Player with Ball: {self. lastPlayerWithBall}")
        # print(f"min_distance: {min_distance}")
        # print(f"Is Ball on possession? {self.ball.inPossession}")
        # print(f"Frames in possession: {self.ball.framesInPossession}")

        # Si hay un jugador cercano a la pelota
        if min_distance < threshold:

            # print(f"Ball data: {self.ball}")

            # Si no hay jugador con pelota
            # Directamente se asigna y empieza el conteo
            if self.lastPlayerWithBall is None:
                print("SOLO DEBE IMPRIMIRSE UNA VEZ CUANDO EL JUGADOR NO ESTÉ ASIGNADO")
                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1

                self.mediumPlayer = closest_player

            # Si el mismo jugador sigue teniendo la pelota
            elif closest_player.id == self.lastPlayerWithBall.id or closest_player.id == self.mediumPlayer.id :
                self.ball.framesInPossession += 1


            # Suponiendo que un jugador llega a la posesion
            # La pelota se aleja
            # Posesion pasa a False
            # Si la pelota llega a otro jugador se verifica que tenga posesion
            # Pero la unica forma de que tenga posesion es que el 

            # Si es diferente jugador y además el closest no es el medium
            elif closest_player.id != self.lastPlayerWithBall.id:
                self.ball.framesInPossession = 1
                self.mediumPlayer = closest_player
                
                # if  self.ball.framesInPossession <= 2:

            
            # Si el jugador mas cercano es diferente al ultimo guardado
            # y ademas ya la tuvo alguien suficiente tiempo
            if closest_player.id != self.lastPlayerWithBall.id and self.ball.framesInPossession > 2:
                print("Inicio de pase")
                durationPass = video_info.frames_to_seconds(self.ball.framesInTransit)
                secondInitPass = video_info.frames_to_seconds(self.ball.initFrameNumber)

                newPass = Pass(initPlayer = self.lastPlayerWithBall, finalPlayer = closest_player, frames = self.ball.framesInTransit, durationPass= durationPass, initFrame = self.ball.initFrameNumber, secondInitPass = secondInitPass)

                print(newPass)

                # Variabe video_info

                if newPass.valid:
                    print("Guardando pase")
                    self.newPass = newPass
                    self.newPass.save_to_csv("passes.csv")
                else:
                    pass

                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1
                # Codigo para agregar un pase
                self.ball.framesInTransit = 0

                self.mediumPlayer = closest_player
            # Cambia de jugador por un instante (de paso)
            # else:
                # self.ball.framesInTransit += 1  # Para saber cuanto tiempo tardó el pase


            self.ball.inPossession = True
        
        # Pelota no cerca de la zona
        else:
            self.ball.inPossession = False

            # Es decir que por lo menos alguien tuvo antes la pelota
            if self.lastPlayerWithBall is not None:
                # self.ball.framesInPossession = 0
                self.ball.framesInTransit += 1  # Para saber cuanto tiempo tardó el pase
                self.ball.initFrameNumber = frame_number
            else:
                pass




            # self.lastPlayerWithBall = None
            # Aqui queda pendiente si pongo framesInPossession = 0
            # Osea que si el balón se aleja de un jugador pero  despues se acerca estuvo en posesion de el siempre? o debe reiniciar el contador?



    
    
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