

def _prepare_ros2_environment():
    """Best-effort to ensure ROS 2 env variables are present for ROS bridge/rclpy."""
    import os, subprocess
    if os.environ.get("RMW_IMPLEMENTATION", "") == "":
        os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    candidate_setups = [
        os.environ.get("ROS_SETUP", ""),
        os.path.expanduser("~/.bashrc_ros2"),
        "/opt/ros/humble/setup.bash",
        "/opt/ros/foxy/setup.bash",
    ]
    for setup in candidate_setups:
        if not setup or not os.path.exists(setup):
            continue
        try:
            out = subprocess.check_output(f"bash -c 'source {setup} && env -0'", shell=True)
            for entry in out.split(b"\x00"):
                if not entry:
                    continue
                k, _, v = entry.partition(b"=")
                os.environ[k.decode()] = v.decode()
            break
        except Exception:
            continue

class ROSInterface:
    """
    Lightweight ROS debug interface for publishing sensors (e.g., tiled cameras).
    Prefer Isaac Sim ROS2 Bridge; fallback to rclpy if bridge is unavailable.
    """

    def __init__(self, env, enable: bool = True, use_ros_bridge_first: bool = True, publish_env_indices: int | list[int] = 0, publish_rate_hz: float = 5.0, save_images: bool = False, save_dir: str = "runs/ros_debug", save_every_n: int = 30):
        self.env = env
        self.enabled = enable
        self.use_ros_bridge_first = use_ros_bridge_first
        if isinstance(publish_env_indices, int):
            self.publish_env_indices = [publish_env_indices]
        else:
            self.publish_env_indices = list(publish_env_indices)
        self.publish_period = 1.0 / max(1e-6, float(publish_rate_hz))
        self._save_images = bool(save_images)
        self._save_dir = str(save_dir)
        self._save_every_n = max(1, int(save_every_n))

        # runtime
        self._last_pub_time = 0.0
        self._mode = None  # "ros_bridge" | "rclpy" | None
        self._bridge_ok = False
        self._rclpy_ok = False
        self._save_counter = 0

        # ros bridge handles
        self._bridge_publishers = {}  # {(cam_name, data_type, env_idx): <publisher>}

        # rclpy handles
        self._rclpy = None
        self._rclpy_node = None
        self._rclpy_publishers = {}  # {(cam_name, data_type, env_idx): publisher}
        # track last published frame per (cam_name, env_idx)
        self._last_cam_frame = {}  # {(cam_name, env_idx): int}

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Try ROS bridge first, then rclpy fallback."""
        if self.use_ros_bridge_first:
            self._bridge_ok = self._try_init_ros_bridge()
            if self._bridge_ok:
                self._mode = "ros_bridge"
                return
            self._rclpy_ok = self._try_init_rclpy()
            self._mode = "rclpy" if self._rclpy_ok else None
        else:
            self._rclpy_ok = self._try_init_rclpy()
            if self._rclpy_ok:
                self._mode = "rclpy"
                return
            self._bridge_ok = self._try_init_ros_bridge()
            self._mode = "ros_bridge" if self._bridge_ok else None

    def shutdown(self):
        if self._mode == "rclpy" and self._rclpy_ok:
            try:
                for pub in self._rclpy_publishers.values():
                    pub.destroy()
                if self._rclpy_node is not None:
                    self._rclpy_node.destroy_node()
                if self._rclpy is not None:
                    self._rclpy.shutdown()
            except Exception:
                pass
        # bridge publishers are owned by extension; nothing explicit here

    # --- ROS bridge path ---
    def _try_init_ros_bridge(self) -> bool:
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            ext_mgr = app.get_extension_manager()
            if not ext_mgr.is_extension_enabled("omni.isaac.ros2_bridge"):
                # try enable on the fly
                from omni.isaac.core.utils.extensions import enable_extension
                enable_extension("omni.isaac.ros2_bridge")
            # simple smoke test import
            import omni.isaac.ros2_bridge  # noqa: F401
            return True
        except Exception:
            return False

    # --- rclpy path ---
    def _try_init_rclpy(self) -> bool:
        # Attempt to ensure ROS 2 env is available inside this process
        self._prepare_ros2_environment()
        try:
            import rclpy
            from rclpy.node import Node  # noqa: F401
            rclpy.init(args=None)
            self._rclpy = rclpy
            self._rclpy_node = rclpy.create_node("isaaclab_debug_publisher")
            return True
        except Exception as e:
            print(f"[ROSInterface] rclpy init failed: {e}")
            return False

    def _prepare_ros2_environment(self):
        """
        Heuristic: if RMW is missing (typical in kit subprocess), try to source a setup.bash.
        User may have sourced their shell, but the embedded python may not inherit it.
        """
        import os
        import subprocess
        # If already has RMW libs, skip
        if os.environ.get("RMW_IMPLEMENTATION", "") == "":
            os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        # If rclpy import fails, try sourcing known setup locations
        candidate_setups = [
            os.environ.get("ROS_SETUP", ""),
            os.path.expanduser("~/.bashrc_ros2"),
            "/opt/ros/humble/setup.bash",
            "/opt/ros/foxy/setup.bash",
        ]
        for setup in candidate_setups:
            if not setup or not os.path.exists(setup):
                continue
            try:
                cmd = f"bash -c 'source {setup} && env -0'"
                output = subprocess.check_output(cmd, shell=True)
                for entry in output.split(b"\x00"):
                    if not entry:
                        continue
                    k, _, v = entry.partition(b"=")
                    os.environ[k.decode()] = v.decode()
                break
            except Exception:
                continue

    # --- Public API ---
    def setup_cameras(self, topic_ns: str = "/debug"):
        if not self.enabled or self._mode is None:
            return
        # Prepare publishers for each configured camera
        for cam_name, cam in getattr(self.env, "_cameras", {}).items():
            for env_idx in self.publish_env_indices:
                # init frame tracker
                try:
                    self._last_cam_frame[(cam_name, env_idx)] = int(cam.frame[env_idx].item())
                except Exception:
                    self._last_cam_frame[(cam_name, env_idx)] = -1
                # Create publishers for all data types present in sensor
                for data_type in cam.cfg.data_types:
                    topic = f"{topic_ns}/{cam_name}/env_{env_idx}/{data_type}"
                    if self._mode == "ros_bridge":
                        self._bridge_publishers[(cam_name, data_type, env_idx)] = self._create_ros_bridge_publisher(topic, data_type)
                    elif self._mode == "rclpy":
                        self._rclpy_publishers[(cam_name, data_type, env_idx)] = self._create_rclpy_publisher(topic, data_type)

    def update(self, sim_time_s: float):
        """Publish at configured rate."""
        if not self.enabled or self._mode is None:
            return
        if (sim_time_s - self._last_pub_time) < self.publish_period:
            return
        self._last_pub_time = sim_time_s

        for cam_name, cam in getattr(self.env, "_cameras", {}).items():
            # publish only if camera advanced a frame (avoid forcing GPU buffer update)
            for env_idx in self.publish_env_indices:
                try:
                    cur_frame = int(cam.frame[env_idx].item())
                except Exception:
                    cur_frame = -1
                last_frame = self._last_cam_frame.get((cam_name, env_idx), -1)
                if cur_frame <= last_frame:
                    continue
                self._last_cam_frame[(cam_name, env_idx)] = cur_frame
                # safe fetch of already-updated buffers
                try:
                    outputs = cam.data.output  # should not trigger heavy GPU update if already updated
                except Exception as e:
                    print(f"[ROSInterface] Skip publish (camera data not ready): {e}")
                    continue
                for data_type in cam.cfg.data_types:
                    try:
                        img = outputs[data_type][env_idx]  # H x W x C
                    except Exception as e:
                        print(f"[ROSInterface] Fetch {cam_name}/{data_type} failed: {e}")
                        continue
                    # optional: dump to disk for debugging
                    self._maybe_save_image(cam_name, env_idx, data_type, img, cur_frame)
                    if self._mode == "ros_bridge":
                        pub = self._bridge_publishers.get((cam_name, data_type, env_idx))
                        if pub is not None:
                            self._publish_via_ros_bridge(pub, img, data_type)
                    elif self._mode == "rclpy":
                        pub = self._rclpy_publishers.get((cam_name, data_type, env_idx))
                        if pub is not None:
                            self._publish_via_rclpy(pub, img, data_type)

    def _maybe_save_image(self, cam_name: str, env_idx: int, data_type: str, img_tensor, frame_id: int):
        if not self._save_images:
            return
        try:
            import os
            import numpy as np
            from PIL import Image
            os.makedirs(self._save_dir, exist_ok=True)
            if (self._save_counter % self._save_every_n) != 0:
                self._save_counter += 1
                return
            self._save_counter += 1
            arr = img_tensor.detach().to("cpu").numpy()
            if data_type == "rgb":
                if arr.dtype != np.uint8:
                    # heuristic scaling
                    maxv = float(arr.max()) if arr.size > 0 else 1.0
                    if maxv <= 1.5:
                        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                out = os.path.join(self._save_dir, f"{cam_name}_env{env_idx}_rgb_f{frame_id}.png")
                img.save(out)
            elif data_type == "depth":
                # save 16-bit depth for readability
                import numpy as np
                depth = arr.astype(np.float32)
                # clip to, e.g., 100 m for visualization
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth = np.clip(depth, 0.0, 100.0)
                depth16 = (depth * 655.35).astype(np.uint16)  # 100 m -> ~65535
                Image.fromarray(depth16, mode="I;16").save(
                    os.path.join(self._save_dir, f"{cam_name}_env{env_idx}_depth_f{frame_id}.png")
                )
        except Exception as e:
            print(f"[ROSInterface] Save image failed: {e}")

    # --- Bridge publish helpers ---
    def _create_ros_bridge_publisher(self, topic: str, data_type: str):
        """
        Minimal publisher binding via ROS2 Bridge. We use generic Image bridge.
        """
        try:
            from omni.isaac.ros2_bridge import _ros2_bridge as ros2_bridge
            from sensor_msgs.msg import Image
            # Real bridge Python API may differ; leave as placeholder (not used by default)
            return None
            return pub
        except Exception as e:
            print(f"[ROSInterface] ROS2 Bridge publisher failed for {topic}: {e}")
            return None

    def _publish_via_ros_bridge(self, pub, img_tensor, data_type: str):
        try:
            import numpy as np
            from sensor_msgs.msg import Image
            msg = Image()
            arr = img_tensor.detach().to("cpu").numpy()
            h, w = arr.shape[0], arr.shape[1]
            msg.height = h
            msg.width = w
            if data_type == "rgb":
                # scale to uint8 if needed
                if arr.dtype != np.uint8:
                    maxv = float(arr.max()) if arr.size > 0 else 1.0
                    if maxv <= 1.5:
                        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                msg.encoding = "rgb8"
                msg.step = w * 3
                msg.data = arr.tobytes()
            elif data_type == "depth":
                msg.encoding = "32FC1"
                msg.step = w * 4
                msg.data = arr.astype(np.float32).tobytes()
            else:
                return
            pub.publish(msg)
        except Exception as e:
            print(f"[ROSInterface] Bridge publish failed: {e}")

    # --- rclpy publish helpers ---
    def _create_rclpy_publisher(self, topic: str, data_type: str):
        try:
            from sensor_msgs.msg import Image
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            qos = QoSProfile(
                # Use RELIABLE to match rviz2/image_view default
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            pub = self._rclpy_node.create_publisher(Image, topic, qos)
            return pub
        except Exception as e:
            print(f"[ROSInterface] rclpy publisher failed for {topic}: {e}")
            return None

    def _publish_via_rclpy(self, pub, img_tensor, data_type: str):
        try:
            import numpy as np
            from sensor_msgs.msg import Image
            from std_msgs.msg import Header
            msg = Image()
            msg.header = Header()
            msg.header.frame_id = "world"
            arr = img_tensor.detach().to("cpu").numpy()
            h, w = arr.shape[0], arr.shape[1]
            msg.height = h
            msg.width = w
            if data_type == "rgb":
                # scale to uint8 if needed
                if arr.dtype != np.uint8:
                    maxv = float(arr.max()) if arr.size > 0 else 1.0
                    if maxv <= 1.5:
                        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                msg.encoding = "rgb8"
                msg.step = w * 3
                msg.data = arr.tobytes()
            elif data_type == "depth":
                msg.encoding = "32FC1"
                msg.step = w * 4
                msg.data = arr.astype(np.float32).tobytes()
            else:
                return
            pub.publish(msg)
            # spin_once not required here; caller controls update cadence
        except Exception as e:
            print(f"[ROSInterface] rclpy publish failed: {e}")
