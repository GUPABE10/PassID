import csv
import os

class Player:
    """
    Represents a player with an associated team (cluster ID).
    
    Parameters:
    - player_id: Unique identifier for the player.
    - team: Cluster ID representing the player's team.
    """
    def __init__(self, player_id: int, team: str):
        self.id = player_id  # Same as the tracked ID
        self.team = team

    def __str__(self):
        return f"Player(player_id={self.id}, team={self.team})"


class Ball:
    """
    Represents the ball in the match, tracking its possession and transit status.
    
    Parameters:
    - id: Unique identifier for the ball.
    """
    def __init__(self, id):
        self.id = id
        self.inPossession = False
        self.framesInPossession = 0  # Number of frames the ball is in possession
        self.framesInTransit = 0  # Frames while the ball is not in possession
        self.initFrameNumber = 0  # Starting frame number when transit starts

    def __str__(self):
        return f"Ball(id={self.id})"
    

class Pass:
    """
    Represents a pass between two players, including validation and timing information.
    
    Parameters:
    - initPlayer: Player initiating the pass.
    - finalPlayer: Player receiving the pass.
    - frames: Total frames from start to end of the pass.
    - secondFinalPass: Timestamp for the end of the pass.
    - initFrame: Initial frame number for the pass.
    - secondInitPass: Timestamp for the start of the pass.
    """
    def __init__(self, initPlayer, finalPlayer, frames, secondFinalPass, initFrame, secondInitPass):
        self.initPlayer = initPlayer
        self.finalPlayer = finalPlayer
        self.frames = frames
        self.secondFinalPass = secondFinalPass
        self.initFrame = initFrame
        self.secondInitPass = secondInitPass
        
        self.valid = self.isValid()  # Determine if pass is valid (from teammate to teammate)

    def isValid(self):
        """
        Check if the pass is valid by verifying if both players are on the same team.
        
        Returns:
        - True if both players are on the same team, False otherwise.
        """
        return self.initPlayer.team == self.finalPlayer.team
    
    def save_to_csv(self, file_path):
        """
        Save the pass information to a CSV file.
        
        Parameters:
        - file_path: Path to the CSV file.
        """
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
            
            # Write header if the file does not exist
            if not file_exists:
                writer.writerow(['Passer (id)', 'Receiver (id)', 'Start Time (s)', 'Duration (s)', 'End Time (s)'])
            
            # Write pass data to CSV
            writer.writerow(row)

    def __str__(self):
        return (f"Pass(initPlayer={self.initPlayer} "
                f"finalPlayer={self.finalPlayer} "
                f"frames={self.frames}, initFrame={self.initFrame}, "
                f"secondInitPass={self.secondInitPass}, secondFinalPass={self.secondFinalPass}, valid={self.valid})")
