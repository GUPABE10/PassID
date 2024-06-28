import csv
import os

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
        self.framesInTransit = 0
        self.initFrameNumber = 0

    def __str__(self):
        return f"Ball(id={self.id})"
    

class Pass:
    def __init__(self, initPlayer, finalPlayer, frames, secondFinalPass, initFrame, secondInitPass):
        self.initPlayer = initPlayer
        self.finalPlayer = finalPlayer
        self.frames = frames
        self.secondFinalPass = secondFinalPass
        self.initFrame = initFrame
        self.secondInitPass = secondInitPass
        
        self.valid = self.isValid() # Esto dice si e sun pase valido de compañero a compañero

    def isValid(self):
        return self.initPlayer.team == self.finalPlayer.team
    
    def save_to_csv(self, file_path):
        # Define the row to be added
        row = [
            self.initPlayer.id,
            self.finalPlayer.id,
            self.secondInitPass,
            self.secondFinalPass - self.secondInitPass,
            self.secondFinalPass
        ]
        
        # Check if the file already exists
        file_exists = os.path.isfile(file_path)
        
        # Open the file in append mode
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header
            if not file_exists:
                writer.writerow(['Passer (id)', 'Receiver (id)', 'Start Time (s)', 'Duration (s)', 'End Time (s)'])
            
            # Write the row of data
            writer.writerow(row)

    def __str__(self):
        return (f"Pass(initPlayer={self.initPlayer} "
                f"finalPlayer={self.finalPlayer}"
                f"frames={self.frames}, initFrame={self.initFrame}, "
                f"secondInitPass={self.secondInitPass}, secondFinalPass={self.secondFinalPass}, valid={self.valid})")