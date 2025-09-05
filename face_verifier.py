from deepface import DeepFace
import warnings
warnings.filterwarnings("ignore")

def verify_faces(img1_path, img2_path):
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="ArcFace",
            detector_backend="retinaface"
        )
        return result.get("verified", False), result.get("distance", None)
    except Exception as e:
        print(f"[DeepFace Error]: {e}")
        return False, None