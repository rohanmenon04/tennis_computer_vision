from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
    
    def detect_frame(self, frame):
        """
        Takes in a single frame from a video and draws a bounding box for the players
        Args:
            frame: a single frame from a video
        :returns: a dictionary where the keys are id's and the values are the bounding boxes
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {} # Key is ID, output is bounding box

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result
        
        return player_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Takes in all the frames of a video and calls the detect frames method to draw all the relevant bounding boxes
        Args:
            frames: a full video converted into frames
        Returns:
            player detections: a list of player_dict objects which contain the id and bounding box for each player for each frame
        """
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as file:
                player_detections = pickle.load(file)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as file:
                pickle.dump(player_detections, file)
        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draws bounding boxes on a video once the player_detections list has been properly created
        Args:
            video_frames: this represents all the video frames of the video
            player_detections: result of the detect_frames function, a list of dictionaries of id's and bounding boxes
        Returns:
            a list of all the video frames, with the bounding boxes drawn
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        players_frame_one = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, players_frame_one)

        filtered_player_detections = []

        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
    
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            # Calculate distance from each player to center of court and see closest

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort distances in ascending order
            
        distances = sorted(distances, key=lambda x:x[1])
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players