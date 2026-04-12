import os
import subprocess
from pathlib import Path
from typing import Sequence, Optional, Tuple

import cv2
import numpy as np

from lib.transform import convert_euler_to_matrix, rotmat2qvec


class RenderImageProcessor:
    """
    Blender-based replacement for the original OSG `RenderImageProcessor`.

    It exposes a compatible API:
      - `update_pose(Trans, Rot, fovy=None)`
      - `get_color_image() -> np.ndarray[h, w, 3] (BGR)`
      - `save_color_image(path: str)`

    Internally it:
      1) Writes a temporary intrinsics file and a pose file (COLMAP style).
      2) Calls Blender in background with `RGB_renderer.py` to render one image.
      3) Loads the rendered image from disk and returns it as a NumPy array.

    NOTE:
      - This is an *offline* style renderer (each call spawns a Blender process),
        so it will be much slower than the original in-process OSG renderer.
      - You MUST make sure the Blender-related paths in the JSON config are
        correctly set, especially:
          config["render2loc"]["blender"] = {
              "blender_path": ".../blender",
              "rgb_project_path": ".../your_rgb_project.blend",
              "python_rgb_path": ".../utils/blender/RGB_renderer.py",
              "origin_xml": ".../origin.xml",
              "sensor_width": ...,
              "sensor_height": ...,
              "f_mm": ...
          }
    """

    def __init__(self, config: dict):
        self.config = config

        render_conf = self.config["render2loc"]
        blender_conf = render_conf["blender"]

        # --- render resolution & intrinsics (in pixels) ---
        render_camera = render_conf["render_camera"]
        self.view_width = int(render_camera[0])
        self.view_height = int(render_camera[1])
        self.fx = float(render_camera[2])
        self.fy = float(render_camera[3])
        self.cx = float(render_camera[4])
        self.cy = float(render_camera[5])

        # --- blender / project paths ---
        self.blender_path = blender_conf["blender_path"]
        self.rgb_project_path = blender_conf["rgb_project_path"]
        self.python_rgb_path = blender_conf["python_rgb_path"]
        self.origin_xml = blender_conf["origin_xml"]

        # camera sensor settings for Blender script
        self.sensor_width_mm = float(blender_conf["sensor_width"])
        self.sensor_height_mm = float(blender_conf["sensor_height"])
        self.f_mm = float(blender_conf["f_mm"])

        # directory where temporary intrinsics / poses / renders are stored
        base_results = render_conf.get("results", "") or "."
        self.temp_dir = Path(base_results) / "blender_realtime_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.intrinsics_path = self.temp_dir / "intrinsics.txt"
        self.poses_path = self.temp_dir / "poses.txt"
        self.images_dir = self.temp_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # write intrinsics file once (COLMAP-like simple txt)
        self._write_intrinsics_file()

        # last pose (world coordinates)
        self._last_translation: Optional[Sequence[float]] = None
        self._last_euler: Optional[Sequence[float]] = None

    # --------------------------------------------------------------------- #
    # public API (OSG-compatible)
    # --------------------------------------------------------------------- #
    def update_pose(
        self,
        Trans: Sequence[float],
        Rot: Sequence[float],
        fovy: Optional[float] = None,
    ) -> None:
        """
        Store the latest camera pose.

        Args:
            Trans: iterable of length 3, camera translation.
            Rot: iterable of length 3, camera Euler angles (in degrees, XYZ order).
            fovy: unused here, kept for API compatibility.
        """
        self._last_translation = list(Trans)
        self._last_euler = list(Rot)

    def get_color_image(self) -> np.ndarray:
        """
        Render the current pose with Blender and return a BGR uint8 image.
        """
        if self._last_translation is None or self._last_euler is None:
            raise RuntimeError("update_pose must be called before get_color_image().")

        img_name = "frame"

        # 1) write a single-pose COLMAP-style extrinsics file
        self._write_single_pose_file(img_name, self._last_translation, self._last_euler)

        # 2) call Blender in background to render this pose
        self._call_blender(img_name)

        # 3) load the rendered image from disk
        img_path = self.images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Expected Blender output image not found: {img_path}")

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read Blender output image: {img_path}")

        # ensure size matches desired resolution
        if image.shape[1] != self.view_width or image.shape[0] != self.view_height:
            image = cv2.resize(image, (self.view_width, self.view_height), interpolation=cv2.INTER_LINEAR)

        return image

    def save_color_image(self, outputs: str) -> None:
        """
        Convenience wrapper: render and save to disk (BGR JPEG/PNG depending on suffix).
        """
        img = self.get_color_image()
        out_path = Path(outputs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _write_intrinsics_file(self) -> None:
        """
        Write a minimal intrinsics file that `RGB_renderer.py` can parse.
        Format: one line, where columns 3..8 are: w h fx fy cx cy
        """
        line = f"image 0 {self.view_width} {self.view_height} {self.fx} {self.fy} {self.cx} {self.cy}\n"
        with self.intrinsics_path.open("w") as f:
            f.write(line)

    def _pose_to_colmap_q_t(
        self, translation: Sequence[float], euler_deg: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert (translation, Euler[deg]) into COLMAP-style (qvec, tvec) of a w2c matrix.

        For consistency we:
          - build c2w matrix from the given translation + Euler (XYZ degrees),
          - invert to get w2c,
          - extract R_w2c and t_w2c directly.
        """
        # c2w rotation from Euler
        R_c2w = convert_euler_to_matrix(euler_deg)
        t_c2w = np.asarray(translation, dtype=np.float64).reshape(3)

        T_c2w = np.eye(4, dtype=np.float64)
        T_c2w[0:3, 0:3] = R_c2w
        T_c2w[0:3, 3] = t_c2w

        T_w2c = np.linalg.inv(T_c2w)
        R_w2c = T_w2c[0:3, 0:3]
        t_w2c = T_w2c[0:3, 3]

        qvec = rotmat2qvec(R_w2c).astype(np.float64)
        tvec = t_w2c.astype(np.float64)
        return qvec, tvec

    def _write_single_pose_file(
        self,
        name: str,
        translation: Sequence[float],
        euler_deg: Sequence[float],
    ) -> None:
        """
        Write a pose file with a single COLMAP-style pose row.
        """
        qvec, tvec = self._pose_to_colmap_q_t(translation, euler_deg)
        values = " ".join(map(str, np.concatenate([qvec, tvec])))
        line = f"{name} {values}\n"
        with self.poses_path.open("w") as f:
            f.write(line)

    def _call_blender(self, img_name: str) -> None:
        """
        Invoke Blender in background mode with RGB_renderer.py to render a single image.
        """
        # Arguments order must match RGB_renderer.py expectations:
        #   xml_path sensor_height sensor_width f_mm intrin_file pose_file image_save_path
        cmd = [
            self.blender_path,
            "-b",
            self.rgb_project_path,
            "-P",
            self.python_rgb_path,
            "--",
            self.origin_xml,
            str(self.sensor_height_mm),
            str(self.sensor_width_mm),
            str(self.f_mm),
            str(self.intrinsics_path),
            str(self.poses_path),
            str(self.images_dir),
        ]

        # Use subprocess with a list of args to handle spaces in paths correctly
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Blender process failed with return code {e.returncode}") from e


