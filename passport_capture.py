import cv2
import numpy as np


class PassportPhotoCapturer:
    """
    Extract a passport photo from a 4-corner quadrilateral using homography + warpPerspective.
    """

    def __init__(
        self,
        output_size=(350, 450),  # (width, height) pixels, 35:45 ratio
        border_mode=cv2.BORDER_REPLICATE,
        interpolation=cv2.INTER_CUBIC,
    ):
        self.output_size = tuple(output_size)
        self.border_mode = border_mode
        self.interpolation = interpolation

    @staticmethod
    def _order_points(pts_xy):
        """
        Ensure consistent ordering: [top-left, top-right, bottom-right, bottom-left].
        """
        pts = np.asarray(pts_xy, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"Expected corners shape (4,2), got {pts.shape}")

        s = pts.sum(axis=1)
        diff = pts[:, 0] - pts[:, 1]

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(diff)]
        bl = pts[np.argmin(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _clip_points_to_frame(pts, frame_shape):
        h, w = frame_shape[:2]
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def extract_passport_photo(self, frame, corners, clip_to_frame=True, return_H=False):
        """
        Warp region defined by `corners` into a passport photo (output_size).

        Returns:
            warped (and optionally homography H)
        """
        src = self._order_points(corners)
        if clip_to_frame:
            src = self._clip_points_to_frame(src, frame.shape)

        out_w, out_h = self.output_size
        dst = np.array(
            [[0, 0],
             [out_w - 1, 0],
             [out_w - 1, out_h - 1],
             [0, out_h - 1]],
            dtype=np.float32
        )

        H = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            frame,
            H,
            (out_w, out_h),
            flags=self.interpolation,
            borderMode=self.border_mode
        )

        if return_H:
            return warped, H
        return warped