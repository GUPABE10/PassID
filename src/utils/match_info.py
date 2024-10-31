import cv2
import numpy as np

from utils.match_objects import Player, Ball, Pass

class VideoInfo:
    def __init__(self, video_path):
        """
        Initialize VideoInfo with the given video path.
        
        Parameters:
        - video_path: Path to the video file.
        """
        self.video_path = video_path
        self.width, self.height, self.duration, self.frame_rate, self.frame_count = self.get_video_info()

    def get_video_info(self):
        """
        Extract video information such as width, height, duration, frame rate, and frame count.
        
        Returns:
        - width: Video width in pixels.
        - height: Video height in pixels.
        - duration: Video duration in seconds.
        - frame_rate: Frame rate of the video.
        - frame_count: Total number of frames in the video.
        """
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
        """
        Convert frame count to seconds.
        
        Parameters:
        - frames: Number of frames.
        
        Returns:
        - Time in seconds.
        """
        if frames < 0 or frames > self.frame_count:
            raise ValueError("Frame number out of range")
        return frames / self.frame_rate

    def __str__(self):
        return f"VideoInfo(frame_rate={self.frame_rate}, width={self.width}, height={self.height}, duration={self.duration})"


class Match:
    def __init__(self):
        """
        Initialize Match with empty player and ball attributes.
        """
        self.players = {}
        self.extras = set()  # Avoid duplicate IDs
        self.ball = None
        self.lastPlayerWithBall = None
        self.newPass = None
        self.mediumPlayer = None  # Temporary player for intermediate possession

        self.initPass = False

    def add_player(self, player_id, team):
        """
        Add a player to the match.
        
        Parameters:
        - player_id: Unique player identifier.
        - team: Team identifier.
        """
        self.players[player_id] = Player(player_id, team)
        
    def add_extra_people(self, extra_id):
        """
        Add an extra person ID to the match.
        
        Parameters:
        - extra_id: Unique identifier for non-player people.
        """
        self.extras.add(extra_id)
        
    def assignBall(self, ball_id):
        """
        Assign a ball object to the match by its ID.
        
        Parameters:
        - ball_id: Unique identifier for the ball.
        """
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
        """
        Check if the background of the bounding box area is green.
        
        Parameters:
        - image: Input image.
        - bbox: Bounding box coordinates.
        - padding: Optional padding around the bounding box.
        
        Returns:
        - True if the background is mostly green, False otherwise.
        """
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

        hsv = cv2.cvtColor(surrounding_region, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_ratio = np.sum(green_mask) / (green_mask.size * 255)

        return green_ratio > 0.2

    def assign_ball_to_match(self, tracked_objects, image, padding=10):
        """
        Assign the ball based on tracked objects and background check.
        
        Parameters:
        - tracked_objects: List of tracked objects.
        - image: Input image for background check.
        - padding: Optional padding for background check.
        """
        ball_candidates = []
        for obj in tracked_objects:
            if obj.label == 2:
                bbox = obj.estimate.flatten().tolist()
                ball_candidates.append(obj.id)
        
        if self.ball is None:
            if len(ball_candidates) == 1:
                self.assignBall(ball_id=ball_candidates[0])
            elif len(ball_candidates) > 1:
                self.assignBall(ball_id=0)
        else:
            current_ball_id = self.ball.id
            if len(ball_candidates) == 1 and ball_candidates[0] != current_ball_id:
                self.ball.id = ball_candidates[0]  # Update ball ID

    def update_ball_possession(self, tracked_objects, video_info, frame_number, out_file):
        """
        Update ball possession and detect passes between players.
        
        Parameters:
        - tracked_objects: List of tracked objects.
        - video_info: Video information object.
        - frame_number: Current frame number.
        - out_file: Output file to save pass details.
        """
        if self.ball is None or self.ball.id == 0:
            return
        
        ball_position = next((obj for obj in tracked_objects if obj.id == self.ball.id), None)
        if ball_position is None:
            self.ball.inPossession = False
            self.ball.framesInTransit += 1
            return
        
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_position.estimate.flatten().tolist()
        ball_center_x = (ball_x1 + ball_x2) / 2
        ball_center_y = (ball_y1 + ball_y2) / 2
        
        closest_player = None
        min_distance = float('inf')
        
        for player_id, player in self.players.items():
            player_position = next((obj for obj in tracked_objects if obj.id == player_id), None)

            if player_position is not None:
                player_x1, player_y1, player_x2, player_y2 = player_position.estimate.flatten().tolist()
                
                player_x = player_x1 if ball_center_x < player_x1 else player_x2
                player_y = player_y2
                
                distance = ((ball_center_x - player_x) ** 2 + (ball_center_y - player_y) ** 2) ** 0.5
                distance_proportional = distance / ((video_info.width ** 2 + video_info.height ** 2) ** 0.5)
                
                if distance_proportional < min_distance:
                    min_distance = distance_proportional
                    closest_player = player
        
        threshold = 0.03

        if min_distance < threshold:
            if self.lastPlayerWithBall is None:
                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1
                self.mediumPlayer = closest_player
            elif closest_player.id == self.lastPlayerWithBall.id or closest_player.id == self.mediumPlayer.id:
                self.ball.framesInPossession += 1
            elif closest_player.id != self.lastPlayerWithBall.id and self.ball.framesInPossession > 3:
                durationPass = video_info.frames_to_seconds(self.ball.framesInTransit)
                secondInitPass = video_info.frames_to_seconds(self.ball.initFrameNumber)
                secondFinalPass = video_info.frames_to_seconds(frame_number)

                newPass = Pass(initPlayer=self.lastPlayerWithBall, finalPlayer=closest_player, frames=self.ball.framesInTransit, secondFinalPass=secondFinalPass, initFrame=self.ball.initFrameNumber, secondInitPass=secondInitPass)

                if newPass.valid:
                    self.newPass = newPass
                    self.newPass.save_to_csv(out_file)

                self.lastPlayerWithBall = closest_player
                self.ball.framesInPossession = 1
                self.ball.framesInTransit = 0
                self.mediumPlayer = closest_player
                self.initPass = False
            self.ball.inPossession = True
        else:
            self.ball.inPossession = False

            if self.lastPlayerWithBall is not None and not self.initPass:
                self.ball.initFrameNumber = frame_number
                self.initPass = True
            self.ball.framesInTransit += 1
