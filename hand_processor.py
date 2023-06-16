import cv2
from dataclasses import dataclass


@dataclass
class Finger:
    x: float
    y: float
    z: float

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class HandProcessor:
    active = False

    def __init__(self, click_threshold=0.05):
        self.click_threshold = click_threshold
        self.right_index_position = None
        self.right_thumb_position = None
        self.index_thumb_distance = None
        self.active = False
        self.is_clicking = False
        self.is_new_click = False

        self.calibrated = False
        self.upper_left_position = None
        self.lower_right_position = None

        self.last_position = None
        self.last_velocity = (0, 0)

    def process_hands(self, hands_result):
        if not hands_result.multi_hand_landmarks:
            self.active = False
            return

        self.active = True

        hand_lms = hands_result.multi_hand_landmarks[0]
        self._fingers_position(hand_lms)
        self._process_click()

    def _fingers_position(self, hand_lms):
        self.right_index_position = Finger(
            hand_lms.landmark[8].x, hand_lms.landmark[8].y, hand_lms.landmark[8].z
        )
        self.right_thumb_position = Finger(
            hand_lms.landmark[4].x, hand_lms.landmark[4].y, hand_lms.landmark[4].z
        )

    def _process_click(self):
        self.is_new_click = False

        self.index_thumb_distance = self._distance(
            self.right_index_position, self.right_thumb_position
        )
        is_clicking = self.index_thumb_distance < self.click_threshold

        if is_clicking and not self.is_clicking:
            self.is_new_click = True

        self.is_clicking = is_clicking

    def _distance(self, finger1, finger2):
        return (
            (finger1.x - finger2.x) ** 2
            + (finger1.y - finger2.y) ** 2
            + (finger1.z - finger2.z) ** 2
        ) ** 0.5

    def set_calibration(self, position):
        if self.calibrated:
            return

        if self.upper_left_position is None:
            self.upper_left_position = position
            return

        if self.lower_right_position is None:
            self.lower_right_position = position
            self.calibrated = True

    def draw_calibration_rectangle(self, image):
        if not self.calibrated:
            return

        h, w, _ = image.shape
        cv2.rectangle(
            image,
            (int(self.upper_left_position.x * w), int(self.upper_left_position.y * h)),
            (
                int(self.lower_right_position.x * w),
                int(self.lower_right_position.y * h),
            ),
            (255, 0, 0),
            2,
        )

    def draw_fingers(self, image):
        h, w, _ = image.shape

        if not self.active:
            return

        if self.is_clicking:
            cv2.circle(
                image,
                (
                    int(self.right_index_position.x * w),
                    int(self.right_index_position.y * h),
                ),
                25,
                (0, 255, 0),
                cv2.FILLED,
            )
            return

        cx, cy = int(self.right_index_position.x * w), int(
            self.right_index_position.y * h
        )
        cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        cx, cy = int(self.right_thumb_position.x * w), int(
            self.right_thumb_position.y * h
        )
        cv2.circle(image, (cx, cy), 25, (100, 0, 255), cv2.FILLED)

    def get_position_in_screen(self, screen_size):
        x_position = self.right_index_position.x
        x_position = (x_position - self.upper_left_position.x) / (
            self.lower_right_position.x - self.upper_left_position.x
        )
        x_position = int(x_position * screen_size[0])

        y_position = self.right_index_position.y
        y_position = (y_position - self.upper_left_position.y) / (
            self.lower_right_position.y - self.upper_left_position.y
        )
        y_position = int(y_position * screen_size[1])

        last_velocity = self.last_velocity
        if self.last_position is not None:
            result, last_velocity = self.smooth_damp(
                self.last_position,
                (x_position, y_position),
                self.last_velocity,
                0.05,
                1000,
                0.1,
            )
        else:
            result = (x_position, y_position)

        self.last_velocity = last_velocity
        self.last_position = result

        return result

    def smooth_damp(
        self, current, target, current_velocity, smooth_time, max_speed, delta_time
    ):
        smooth_time = max(0.0001, smooth_time)
        omega = 2.0 / smooth_time

        x = omega * delta_time
        exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

        distance = [target[0] - current[0], target[1] - current[1]]
        max_speed = max_speed * smooth_time

        damp = [max_speed * smooth_time, max_speed * smooth_time]
        current_velocity = [current_velocity[0] * exp, current_velocity[1] * exp]
        result = [
            current[0] + distance[0] * damp[0],
            current[1] + distance[1] * damp[1],
        ]
        result[0] = max(target[0] - max_speed, min(result[0], target[0] + max_speed))
        result[1] = max(target[1] - max_speed, min(result[1], target[1] + max_speed))

        if (target[0] - current[0] > 0.0) == (result[0] > target[0]):
            result[0] = target[0]
            current_velocity[0] = 0.0

        if (target[1] - current[1] > 0.0) == (result[1] > target[1]):
            result[1] = target[1]
            current_velocity[1] = 0.0

        return result, current_velocity
