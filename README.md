# PassID: A Modular System for Pass Detection with Integrated Player Identification in Football

## 1. Introduction

**PassID** is a modular system designed to detect passes and associate them with the players involved in football matches. Using match videos, the system identifies players, detects pass events, and generates a file recording the time each pass occurs along with the identifiers of the involved players.

[![Watch the demonstration video](https://img.youtube.com/vi/q8rvoNvO0mk/0.jpg)](https://youtu.be/q8rvoNvO0mk)

The motivation behind developing **PassID** stems from the need for an automated tool that can accomplish this task without human intervention, following the example of systems proposed in previous research (Morra et al., Theagarajan et al., and the Tryolabs project). However, our system takes it a step further by integrating player identification, team detection, and a pass detection model.


The motivation behind developing **PassID** stems from the need for an automated tool that can accomplish this task without human intervention, following the example of systems proposed in previous research (Morra et al., Theagarajan et al., and the Tryolabs project). However, our system takes it a step further by integrating player identification, team detection, and a pass detection model.

This README describes the system architecture, each of its key modules, and how they interact to achieve accurate pass event detection in football. **PassID** is intended for sports analysts, developers, and researchers interested in applying computer vision techniques to sports analysis.

## 2. System Design

The architecture of **PassID** consists of five main phases, each responsible for an essential task to achieve pass detection and player identification on the field. The modular structure allows each phase to be optimized independently, facilitating the integration of improvements or adjustments across the entire system. The figure below illustrates the system's architecture:

![Modular architecture of the complete pass detection system. The system is divided into two main components: Pass Event Detection and Player Identification, which work together through object detection (OD), multiple object tracking (MOT), and team identification to accurately detect and record passes during a soccer match.](./ch3_systemDesign.png)

### Description of Main Phases

1. **Player and Ball Detection**: This phase employs an object detection (OD) model to locate players and the ball in each video frame. Accuracy in detection is essential, as the results of this phase feed into subsequent stages of the system.

2. **Player and Ball Tracking**: Using data obtained in the detection phase, the system applies a multiple object tracking (MOT) model. This model assigns unique identifiers to each detected player, enabling tracking of their movements throughout the match and maintaining consistent identifiers across frames.

3. **Team Identification**: To detect passes between two players on the same team, the system requires distinguishing between players from different teams. This distinction is primarily based on the players’ uniforms. An unsupervised clustering model is used to group players into two distinct teams, leveraging visual features extracted from the players. This phase is crucial for determining teammate relationships in pass events.

4. **Pass Detection Model**: This module uses information provided by the MOT and team identification models to detect pass events. The approach considers a pass as a sequence of events, using heuristic rules to identify when a pass begins and ends. Specifically, the pass is detected through two fundamental ball states: possession and movement. Possession is defined by proximity between a player and the ball, while movement represents a transitional state where no player has possession.

5. **Output File Generation**: Once pass events are detected, the system generates an output file in CSV format that includes the timestamp of each pass as well as the identifiers of the players involved. This output file enables sports analysts to perform detailed studies on passing patterns and game dynamics.

### Modular Architecture

**PassID** has been designed with a modular structure, allowing each component to function independently and be replaced or updated without affecting the operation of other modules. This modularity not only facilitates future optimizations but also allows for testing different detection or tracking models without altering the main system logic.

This flexibility makes **PassID** an adaptable system, suitable for experimenting with other algorithms or applications across different sports. Additionally, the system is implemented so that detection and tracking models can be customized to meet the user's specific requirements.

The following chapters describe each of these phases in detail, covering model selection and the evaluation process. **PassID** represents a significant contribution to the field of automated sports analysis, establishing a strong foundation for future research and improvements in event detection in football.

## Installation

To set up the environment for running `PassID: A Modular System for Pass Detection with Integrated Player Identification in Football`, please follow these instructions:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/PassID.git
    cd PassID
    ```

2. **Install Requirements**

    Install the required Python libraries by using the following command:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file includes essential libraries, such as:

    - `torch` and `torchvision`: for loading and running deep learning models.
    - `opencv-python`: for processing videos and images.
    - `ultralytics`: for YOLO-based object detection.
    - `hdbscan`: for clustering players to identify team associations.
    - `detectron2` and `norfair`: specialized libraries for object detection and tracking.

    > **Note**: `detectron2` may require additional setup depending on your system’s CUDA version. Refer to the [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) if needed.

## Usage

The `PassID` system offers several functionalities that can be executed as distinct tasks, including `track`, `player_classification`, and `pass_detection`. Each task is designed to fulfill a specific aspect of the overall system, such as tracking, player team identification, or pass detection. To use these tasks, run the main script with the desired task and arguments.

1. **Tracking Task**

    Use this command to track players and the ball in a video or a sequence of images. This will generate visual tracking outputs or an evaluation file depending on the options set.

    ```bash
    python main.py track --files <path/to/video/or/image/folder> \
                         --conf-threshold 0.8 \
                         --track-points bbox \
                         --distance-threshold 1.0 \
                         --distance-function scalar \
                         --backbone resnet50v2 \
                         --draw False \
                         --evalFile False \
                         --isVideo False \
                         --device cuda:0 \
                         --detector FasterRCNN_pretrained \
                         --outputDir outputFiles
    ```

    - `--files`: Path to the video file or folder of images.
    - `--conf-threshold`: Confidence threshold for object detection (default: 0.8).
    - `--track-points`: Tracking points used (`centroid` or `bbox`, default: `bbox`).
    - `--distance-threshold`: Distance threshold for tracking (default: 1.0).
    - `--distance-function`: Distance function for tracking (`scalar` or `iou`, default: `scalar`).
    - `--backbone`: Backbone model for the object detector (e.g., `resnet50v2`).
    - `--draw`: Set to `True` to generate a video with visual tracking (default: `False`).
    - `--evalFile`: Set to `True` to generate an evaluation file (default: `False`).
    - `--isVideo`: Set to `True` if the input is a video (default: `False`).
    - `--device`: CUDA device to use, e.g., `cuda:0` (default: `cuda:0`).
    - `--detector`: Object detector model to use (e.g., `FasterRCNN_pretrained`).
    - `--outputDir`: Directory for saving outputs in evaluation mode.

2. **Player Classification Task**

    This task identifies the team each player belongs to based on the visual characteristics of the player's uniform.

    ```bash
    python main.py player_classification --file <path/to/image>
    ```

    - `--file`: Path to the input image for team clustering.

3. **Pass Detection Task**

    The main functionality of the system, `pass_detection`, detects passes between players in a soccer match. This task processes either a video or a sequence of images and outputs a CSV file with details of each detected pass.

    ```bash
    python main.py pass_detection --files <path/to/video/or/image/folder> \
                                  --conf-threshold 0.8 \
                                  --track-points bbox \
                                  --distance-threshold 0.04 \
                                  --distance-function scalar \
                                  --backbone resnet50v2 \
                                  --isVideo False \
                                  --device cuda:0 \
                                  --detector FasterRCNN_pretrained \
                                  --testMode False
    ```

    - `--files`: Path to the video file or folder of images.
    - `--conf-threshold`: Confidence threshold for object detection (default: 0.8).
    - `--track-points`: Tracking points used (`centroid` or `bbox`, default: `bbox`).
    - `--distance-threshold`: Distance threshold for tracking (default: 0.04).
    - `--distance-function`: Distance function for tracking (`scalar` or `iou`, default: `scalar`).
    - `--backbone`: Backbone model for the object detector (e.g., `resnet50v2`).
    - `--isVideo`: Set to `True` if the input is a video (default: `False`).
    - `--device`: CUDA device to use, e.g., `cuda:0` (default: `cuda:0`).
    - `--detector`: Object detector model to use (e.g., `FasterRCNN_pretrained`).
    - `--testMode`: Set to `True` to enable test mode for debugging and saving intermediate images (default: `False`).

### Example Usage

1. **Tracking a Video File**

    ```bash
    python main.py track --files "data/match_video.mp4" --isVideo True --draw True
    ```

2. **Classifying Players by Team**

    ```bash
    python main.py player_classification --file "data/team_image.jpg"
    ```

3. **Detecting Passes in a Video File**

    ```bash
    python main.py pass_detection --files "data/match_video.mp4" --isVideo True --conf-threshold 0.75
    ```

These commands initialize the `PassID` system to perform tracking, player classification, or pass detection, depending on the chosen task and settings. The output, such as CSV files detailing each pass, visual tracking video, or classification results, will be saved to the specified directories.

## Output Files

After running tasks in the `PassID` system, several output files will be generated based on the specified settings. These files contain essential information for post-processing or analysis.

### Tracking Task

If the `track` task is run with `--evalFile` set to `True`, an evaluation file will be created in the `outputFiles` directory (or the specified output directory). The output may include:

- **Tracking Video** (optional): A video with visual tracking (if `--draw` is `True`).
- **Evaluation File**: Contains tracking data with unique player IDs and their respective positions in each frame.

### Player Classification Task

For the `player_classification` task, **PassID** outputs the player clusters, differentiating teams based on visual features. The outputs include:

- **Classification Image**: An image highlighting players with distinct colors to represent team classification (if visualization is enabled).

### Pass Detection Task

The primary output of the `pass_detection` task is a CSV file, structured to document each detected pass, including timestamps and player identifiers. The generated file contains the following columns:

- **Passer (id)**: The unique identifier of the player initiating the pass.
- **Receiver (id)**: The unique identifier of the player receiving the pass.
- **Start Time (s)**: Timestamp when the pass begins.
- **Duration (s)**: Total duration of the pass.
- **End Time (s)**: Timestamp when the pass ends.

> **Note**: The CSV files generated from pass detection can be used for further statistical analysis or to track game dynamics based on passing sequences.

By providing these outputs, **PassID** delivers valuable data for sports analysts and developers, aiding in analyzing team strategies and player performance.


## About the Project

**PassID: A Modular System for Pass Detection with Integrated Player Identification in Football** was developed as part of a master’s thesis in Computational Sciences at the Instituto Tecnológico y de Estudios Superiores de Monterrey (ITESM), Campus Monterrey. The project seeks to address the challenges of pass detection and player identification in football using video analysis, contributing to the growing field of sports analytics.

This work builds on established methodologies in object detection and tracking, while introducing a unique approach to automate the detection of passes and identify players in real-world football scenarios. **PassID** is designed to be modular and adaptable, allowing future researchers or developers to extend its capabilities or apply it to other sports.

### Author

This project was created by **Benjamín Gutiérrez Padilla** as a part of his master’s thesis. His research interests include computer vision, sports analytics, and machine learning, with a focus on practical applications that push the boundaries of automated event detection. For questions or to discuss potential collaborations, feel free to reach out at [gupabe10@gmail.com].

Thank you for your interest in **PassID**! We hope this tool contributes meaningfully to sports research and inspires future advancements in automated sports analysis.
