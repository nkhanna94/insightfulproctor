
from face_verifier import verify_faces
import cv2
import numpy as np
import requests
import json
import mediapipe as mp
from ultralytics import YOLO
import math
import warnings
warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh

class ProctoringAnalyzer:
    def __init__(self):
        self.yolo_model = YOLO('yolov8m.pt')
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2)
        
        self.gadget_classes = ['laptop', 'tv', 'mouse', 'keyboard', 'remote', 'tablet', 'smartwatch']
        self.phone_classes = ['cell phone']
        self.screen_classes = ['tv', 'laptop', 'monitor']
        self.book_classes = ['book', 'notebook', 'binder', 'folder', 'paper']
        
        self.cosine_threshold = 0.4
        self.gaze_tolerance = 0.24
        self.yaw_max_deg = 30.0
        self.pitch_max_deg = 40.0

    def load_image(self, source):
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, timeout=10)
            img_array = np.frombuffer(response.content, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.imread(source)

    def normalize_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def detect_faces_mediapipe(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                faces.append((x, y, width, height, confidence))
        
        return len(faces), faces

    def interpret_gaze(self, left_offset, right_offset, tolerance=0.24):
        directions = []
        if left_offset[0] > tolerance and right_offset[0] > tolerance:
            directions.append("left")
        elif left_offset[0] < -tolerance and right_offset[0] < -tolerance:
            directions.append("right")
        else:
            directions.append("center")
        if left_offset[1] > tolerance and right_offset[1] > tolerance:
            directions.append("up")
        elif left_offset[1] < -tolerance and right_offset[1] < -tolerance:
            directions.append("down")
        else:
            directions.append("center")
        if directions[0] == "center" and directions[1] == "center":
            return "gaze straight"
        elif directions[0] != "center" and directions[1] == "center":
            return f"gaze {directions[0]}"
        elif directions[0] == "center" and directions[1] != "center":
            return f"gaze {directions[1]}"
        else:
            return f"gaze {directions[0]} and {directions[1]}"

    def detect_gaze_direction(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            return "gaze undetectable"

        landmarks = results.multi_face_landmarks[0].landmark
        left_iris_idx = [473, 474, 475, 476]
        right_iris_idx = [468, 469, 470, 471]
        left_eye_idx = [33, 133, 159, 145]
        right_eye_idx = [263, 362, 386, 374]

        try:
            def get_center(indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                return points.mean(axis=0)

            def get_bounds(indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                return points.min(axis=0), points.max(axis=0)

            left_iris_center = get_center(left_iris_idx)
            right_iris_center = get_center(right_iris_idx)
            left_eye_min, left_eye_max = get_bounds(left_eye_idx)
            right_eye_min, right_eye_max = get_bounds(right_eye_idx)

            def calculate_offset(iris_center, eye_min, eye_max):
                eye_center = (eye_min + eye_max) / 2
                eye_range = (eye_max - eye_min)
                eye_range[eye_range == 0] = 1e-6
                return (iris_center - eye_center) / eye_range  

            left_offset = calculate_offset(left_iris_center, left_eye_min, left_eye_max)
            right_offset = calculate_offset(right_iris_center, right_eye_min, right_eye_max)
            return self.interpret_gaze(left_offset, right_offset, tolerance=self.gaze_tolerance)
        except (IndexError, ZeroDivisionError):
            return "gaze undetectable"
    
    def interpret_head_pose(self, yaw, pitch):
        if yaw is None or pitch is None:
            return "Cannot determine head position"

        STRAIGHT_THRESH = 10
        SLIGHT_THRESH = 20

        if abs(yaw) < STRAIGHT_THRESH:
            yaw_desc = "head straight"
        elif yaw < -SLIGHT_THRESH:
            yaw_desc = "head sharply right"
        elif yaw < 0:
            yaw_desc = "head slightly right"
        elif yaw > SLIGHT_THRESH:
            yaw_desc = "head sharply left"
        else:
            yaw_desc = "head slightly left"

        if abs(pitch) < STRAIGHT_THRESH:
            pitch_desc = "looking straight"
        elif pitch < -SLIGHT_THRESH:
            pitch_desc = "looking sharply up"
        elif pitch < 0:
            pitch_desc = "looking slightly up"
        elif pitch > SLIGHT_THRESH:
            pitch_desc = "looking sharply down"
        else:
            pitch_desc = "looking slightly down"

        return f"{yaw_desc}, {pitch_desc}"


    def detect_head_pose(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return 0, None, None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        
        try:
            image_points = np.array([
                (landmarks[1].x*w,   landmarks[1].y*h),
                (landmarks[152].x*w, landmarks[152].y*h),
                (landmarks[33].x*w,  landmarks[33].y*h),
                (landmarks[263].x*w, landmarks[263].y*h),
                (landmarks[61].x*w,  landmarks[61].y*h),
                (landmarks[291].x*w, landmarks[291].y*h)
            ], dtype="double")
        except IndexError:
            return 0, None, None

        model_points = np.array([
            (0.0,    0.0,    0.0),
            (0.0,  -63.6,  -12.5),
            (-43.3, 32.7,  -26.0),
            (43.3,  32.7,  -26.0),
            (-28.9,-28.9,  -24.1),
            (28.9, -28.9,  -24.1)
        ], dtype="double")

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix,
                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return 0, None, None
        except cv2.error:
            return 0, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

        looking_forward = 1 if abs(yaw) < self.yaw_max_deg and abs(pitch) < self.pitch_max_deg else 0
        
        return looking_forward, round(yaw, 2), round(pitch, 2)
    
    def detect_hands_and_gestures(self, img, detections):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_img)

        hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
        h, w, _ = img.shape
        hand_bboxes = []
        gestures = []
        hands_holding_obj = False

        def get_hand_bbox(hand_landmarks):
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            return [min(xs), min(ys), max(xs), max(ys)]

        def classify_gesture(hand_landmarks):
            tip_ids = [4, 8, 12, 16, 20]
            mcp_ids = [2, 5, 9, 13, 17]
            closed_fingers = 0
            for tip, mcp in zip(tip_ids, mcp_ids):
                tip_lm = hand_landmarks.landmark[tip]
                mcp_lm = hand_landmarks.landmark[mcp]
                dist = ((tip_lm.x - mcp_lm.x) ** 2 + (tip_lm.y - mcp_lm.y) ** 2) ** 0.5
                if dist < 0.07:
                    closed_fingers += 1
            if closed_fingers >= 3:
                return "holding"
            elif closed_fingers == 0:
                return "open"
            else:
                return "unknown"

        def bbox_overlap(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            return (xA < xB) and (yA < yB)

        for hand_landmarks in hand_landmarks_list:
            hand_bbox = get_hand_bbox(hand_landmarks)
            gesture = classify_gesture(hand_landmarks)
            gestures.append(gesture)
            hand_bboxes.append(hand_bbox)

            for det in detections:
                obj_bbox = det['bbox']
                if bbox_overlap(hand_bbox, obj_bbox):
                    hands_holding_obj = True

        sus_gesture = any(g == "unknown" for g in gestures)

        return {
            'num_hands': len(hand_landmarks_list),
            'hands_holding_obj': hands_holding_obj,
            'sus_gesture': sus_gesture,
            'gestures': gestures
        }

    def detect_objects_yolo(self, img):
        results = self.yolo_model(img, conf=0.3, iou=0.5, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = self.yolo_model.names[int(cls)]
                confidence = float(conf)
                bbox = box.cpu().numpy()
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return detections

    def analyze_face_quality(self, img, faces, detections=None):
        if len(faces) == 0:
            return 0, 0  # No faces detected

        h_img, w_img = img.shape[:2]

        for x, y, w, h, confidence in faces:
            # Ignore very small faces
            if w < 60 or h < 60:
                continue

            # Check if face is fully inside image boundaries with a margin
            margin = 10
            if x < margin or y < margin or (x + w) > (w_img - margin) or (y + h) > (h_img - margin):
                return 1, 0  # Face not fully visible

            # Crop face ROI and convert color space for face mesh processing
            face_roi = img[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)

            # If no landmarks detected, face is probably obscured or bad quality
            if not results.multi_face_landmarks:
                return 1, 0

            # Check key landmarks presence (eyes, nose, mouth)
            landmarks = results.multi_face_landmarks[0].landmark
            key_indices = [33, 133, 362, 263, 1, 2, 13, 14]  # eyes, nose, mouth
            try:
                key_points = [(landmarks[i].x, landmarks[i].y) for i in key_indices]
            except Exception:
                return 1, 0

            # If we get here, face is fully visible and landmarks are detected
            return 1, 1

        # If no valid faces found after filtering, return low quality
        return 1, 0

    def count_gadgets(self, detections):
        gadget_count = 0
        phone_present = 0
        second_screen_present = 0
        
        high_conf_gadgets = set()
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name in self.phone_classes and confidence > 0.4:
                phone_present = 1
                if confidence > 0.6:
                    high_conf_gadgets.add('phone')
            
            elif class_name in self.screen_classes and confidence > 0.4:
                second_screen_present = 1
                if confidence > 0.6:
                    high_conf_gadgets.add(class_name)
            
            elif class_name in self.gadget_classes and confidence > 0.4:
                if confidence > 0.6:
                    high_conf_gadgets.add(class_name)
        
        gadget_count = len(high_conf_gadgets)
        
        return min(gadget_count, 6), phone_present, second_screen_present, list(high_conf_gadgets)

    def detect_printed_materials(self, detections):
        for detection in detections:
            if detection['class'] in self.book_classes and detection['confidence'] > 0.25:
                return 1
        return 0

    def generate_caption(self, faces_count, gadgets_count, phone_present, screen_present, 
                        books_present, gadgets, same_person, head_pose_forward, gaze_status, 
                        head_pose_desc, hand_result, yaw_angle=None, pitch_angle=None):

        summary_parts = []
        violations = []

        # Face count
        if faces_count == 0:
            return "No person detected in the image.", "No person detected in the image."
        elif faces_count == 1:
            summary_parts.append("One person detected.")
        else:
            violations.append(f"{faces_count} people detected.")

        # Identity match
        if same_person:
            summary_parts.append("Person matches the reference image.")
        else:
            violations.append("Person does not match the reference image.")

        # Head pose
        if head_pose_forward:
            summary_parts.append(f"Head is facing forward ({head_pose_desc}).")
        else:
            violations.append(f"Head is not facing forward ({head_pose_desc}).")

        # Gaze direction
        if gaze_status and gaze_status.lower() == "gaze straight":
            summary_parts.append("Eyes are looking straight.")
        else:
            violations.append(f"Gaze direction is off: {gaze_status or 'Unknown'}.")

        # Devices & materials
        if phone_present:
            violations.append("Phone detected in the image.")
        if screen_present:
            violations.append("Second screen detected.")
        if books_present:
            violations.append("Printed materials detected.")

        if gadgets_count > 0 and gadgets:
            gadget_summary = ", ".join([g for g in gadgets if g != 'phone'][:3])
            if gadget_summary:
                violations.append(f"Other gadgets detected: {gadget_summary}.")

        if hand_result.get("hands_holding_obj"):
            violations.append("Hands are holding an object.")
        if hand_result.get("sus_gesture"):
            violations.append("Suspicious hand gesture detected.")

        summary_text = "\n".join(summary_parts)
        print("Violations list:", violations)

        if violations:
            violations = "- " + "\n- ".join(violations)
        else:
            violations = ""
    
        return summary_text, violations

    def analyze_dual(self, ref_image_source, image_source):
        try:
            ref_img = self.load_image(ref_image_source)
            img = self.load_image(image_source)

            if img is None:
                return {"error": f"Failed to load proctoring image from {image_source}"}
            if ref_img is None:
                return {"error": f"Failed to load reference image from {ref_image_source}"}

            normalized_ref = self.normalize_image(ref_img)
            normalized_img = self.normalize_image(img)

            is_same, similarity = verify_faces(ref_image_source,image_source)

            num_faces, faces = self.detect_faces_mediapipe(normalized_img)
            detections = self.detect_objects_yolo(normalized_img)

            violation_bboxes = []
            for det in detections:
                cls = det['class']
                conf = det['confidence']
                bbox = det['bbox']
                if (cls in self.phone_classes and conf > 0.25) or \
                   (cls in self.screen_classes and conf > 0.4) or \
                   (cls in self.book_classes and conf > 0.25) or \
                   (cls in self.gadget_classes and conf > 0.6):
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    violation_bboxes.append({'class': cls, 'bbox': bbox, 'confidence': conf}) 

            for (x, y, w, h, conf) in faces:
                violation_bboxes.append({'class': 'face', 'bbox': [x, y, x+w, y+h], 'confidence': conf})

            results = self.face_mesh.process(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = normalized_img.shape[:2]

                left_eye_indices = [33, 133, 159, 145]
                right_eye_indices = [263, 362, 386, 374]

                def get_bbox(indices):
                    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    return [int(x1), int(y1), int(x2), int(y2)]

                left_eye_bbox = get_bbox(left_eye_indices)
                right_eye_bbox = get_bbox(right_eye_indices)

                violation_bboxes.append({'class': 'left_eye', 'bbox': left_eye_bbox, 'confidence': 1.0})
                violation_bboxes.append({'class': 'right_eye', 'bbox': right_eye_bbox, 'confidence': 1.0})

            head_pose_forward, yaw_angle, pitch_angle = self.detect_head_pose(normalized_img)
            
            head_pose_desc = self.interpret_head_pose(yaw_angle, pitch_angle)
            gaze_status = self.detect_gaze_direction(normalized_img)
            
            gadget_count, phone_present, second_screen_present, gadget_list = self.count_gadgets(detections)
            printed_material_present = self.detect_printed_materials(detections)
            face_visible, face_properly_visible = self.analyze_face_quality(normalized_img, faces, detections)

            hand_result = self.detect_hands_and_gestures(img, detections)
            
            summary_text, violations = self.generate_caption(
                    num_faces, gadget_count, phone_present, second_screen_present, 
                    printed_material_present, gadget_list, is_same, head_pose_forward,
                    gaze_status, head_pose_desc, hand_result, yaw_angle, pitch_angle
            )
       
            result = {
                "number_of_faces": num_faces,
                "number_of_gadgets": gadget_count,
                "phone_present": phone_present,
                "second_screen_present": second_screen_present,
                "printed_material_present": printed_material_present,
                "face_visible": face_visible,
                "face_properly_visible": face_properly_visible,
                "same_person_as_reference": is_same,
                "head_pose_forward" : head_pose_forward,
                "gaze_status": gaze_status,
                "head_position": head_pose_desc,
                "hand_gestures":hand_result,
            }
            
            if similarity is not None:
                similarity_percentage = max(0, (1 - similarity) * 100)
                result["face_similarity_percentage"] = round(similarity_percentage, 1)
            
            if gadget_count >= 1:
                result["gadgets"] = gadget_list
            
            result["caption"] = summary_text
            result["violations"] = violations
            result['violation_bboxes'] = violation_bboxes

            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

def main():
    analyzer = ProctoringAnalyzer()
    
    print("Proctoring Analyzer")

    while True:
        print("\n" + "="*50)

        ref_input = str(input("Enter reference image URL/path: ").strip())
        if ref_input.lower() == 'quit':
            break

        current_input = str(input("Enter image URL/path: ").strip())

        if current_input.lower() == 'quit':
            break
        
        print("\nAnalyzing...")
        result = analyzer.analyze_dual(ref_input, current_input)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()