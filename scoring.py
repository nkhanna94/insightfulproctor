from datetime import datetime

class InterviewMonitor:
    def __init__(self):
        self.score = 100
        self.card_history = []
        self.last_amber_time = None

    def evaluate_card(self, result, current_time=None):
        current_time = current_time or datetime.now()

        if result.get("number_of_faces", 0) != 1:
            return self.issue_card("Red 🔴", current_time, "Invalid number of people detected")

        if result.get("same_person_as_reference") is not True:
            return self.issue_card("Red 🔴", current_time, "Different person from reference")

        if not result.get("face_visible", True):
            return self.issue_card("Red 🔴", current_time, "Face not visible")

        if result.get("phone_present") or result.get("second_screen_present") or result.get("printed_material_present"):
            return self.issue_card("Red 🔴", current_time, "Prohibited object detected")

        # Amber card conditions - minor issues
        amber_reasons = []

        if not result.get("head_pose_forward", True):
            amber_reasons.append("Head not facing forward")

        if result.get("gaze_status", "").lower() != "gaze straight":
            amber_reasons.append("Gaze not straight")

        if not result.get("face_properly_visible", True):
            amber_reasons.append("Face partially obscured")

        if result.get("number_of_gadgets", 0) in [1, 2]:
            amber_reasons.append("Minor gadgets visible")

        if amber_reasons:
            return self.issue_card("Amber 🟡", current_time, "; ".join(amber_reasons))

        # Green card - ideal conditions
        return self.issue_card("Green 🟢", current_time, "All conditions ideal")

    def issue_card(self, card_type, current_time, reason=""):
        delta = 0
        if card_type == "Amber 🟡":
            delta = -20
        elif card_type == "Red 🔴":
            delta = -50

        self.score += delta
        self.score = max(0, self.score)

        entry = {
            "card": card_type,
            "timestamp": current_time.isoformat(),
            "reason": reason,
            "score_delta": delta,
            "current_score": self.score
        }

        self.card_history.append(entry)
        return entry

    def get_history(self):
        return self.card_history

    def reset(self):
        self.score = 100
        self.card_history = []
        self.last_amber_time = None