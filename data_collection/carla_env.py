import carla
import numpy as np
import math
import time
import random


class CarlaEnv:
    """
    CARLA environment wrapper for off-road data collection.
    Connects to CARLA server, loads Town07, spawns vehicle,
    attaches sensors, and provides step/reset interface.
    """

    def __init__(
        self,
        host='localhost',
        port=2000,
        town='Town07',
        vehicle_filter='vehicle.jeep.wrangler_rubicon',
        image_width=224,
        image_height=224,
        fps=10,
        seed=42
    ):
        self.host = host
        self.port = port
        self.town = town
        self.vehicle_filter = vehicle_filter
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        self.seed = seed

        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None

        # State tracking
        self.latest_image = None
        self.collision_occurred = False
        self.collision_count = 0
        self.prev_location = None

        # Connect on init
        self._connect()

    def _connect(self):
        """Connect to CARLA server and load map."""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)

        print(f"Loading {self.town}...")
        self.world = self.client.load_world(self.town)
        time.sleep(3)

        # Set fixed timestep for reproducibility
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # Set weather
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        print("CARLA connected and configured.")

    def _spawn_vehicle(self):
        """Spawn ego vehicle at random spawn point."""
        blueprint_library = self.world.get_blueprint_library()

        vehicle_bps = blueprint_library.filter(self.vehicle_filter)
        if not vehicle_bps:
            vehicle_bps = [
                bp for bp in blueprint_library.filter('vehicle.*')
                if int(bp.get_attribute('number_of_wheels')) == 4
            ]
        vehicle_bp = random.choice(vehicle_bps)

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(
                    vehicle_bp, spawn_point
                )
                break
            except Exception:
                continue

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")

        print(f"Spawned: {self.vehicle.type_id}")

    def _attach_sensors(self):
        """Attach front RGB camera and collision sensor."""
        blueprint_library = self.world.get_blueprint_library()

        # Front RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        self.camera.listen(self._on_image)

        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._on_collision)

    def _on_image(self, image):
        """Callback: convert CARLA image to numpy RGB array."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3].copy()

    def _on_collision(self, event):
        """Callback: flag collision event."""
        self.collision_occurred = True
        self.collision_count += 1

    def _get_state(self):
        """
        Build state vector:
        [x, y, z, yaw, vx, vy, vz, speed]
        """
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        speed = math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )

        state_vector = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.yaw,
            velocity.x,
            velocity.y,
            velocity.z,
            speed
        ], dtype=np.float32)

        return state_vector, self.latest_image

    def _compute_reward(self):
        """
        Reward function:
        + forward progress
        - collision penalty
        +/- speed reward
        - smoothness penalty
        """
        reward = 0.0

        transform = self.vehicle.get_transform()
        location = transform.location

        # 1. Forward progress
        if self.prev_location is not None:
            distance_travelled = location.distance(self.prev_location)
            reward += distance_travelled * 2.0

        self.prev_location = location

        # 2. Collision penalty
        if self.collision_occurred:
            reward -= 50.0
            self.collision_occurred = False

        # 3. Speed reward
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        ) * 3.6

        if speed < 2.0:
            reward -= 1.0
        elif speed > 5.0:
            reward += 0.5

        # 4. Smoothness penalty
        control = self.vehicle.get_control()
        reward -= abs(control.steer) * 0.5

        return float(reward)

    def _get_action(self):
        """Get current vehicle control as [steer, throttle, brake]."""
        control = self.vehicle.get_control()
        return np.array(
            [control.steer, control.throttle, control.brake],
            dtype=np.float32
        )

    def reset(self, weather=None):
        """Reset environment and return initial state."""
        self._cleanup()

        if weather is not None:
            self.world.set_weather(weather)

        self.collision_occurred = False
        self.collision_count = 0
        self.prev_location = None
        self.latest_image = None

        self._spawn_vehicle()
        self._attach_sensors()
        self.vehicle.set_autopilot(True)

        for _ in range(5):
            self.world.tick()
        time.sleep(0.1)

        state_vector, image = self._get_state()
        return state_vector, image

    def step(self):
        """
        Tick world one step.
        Returns: state_vector, image, reward, done
        """
        self.world.tick()

        state_vector, image = self._get_state()
        reward = self._compute_reward()
        done = self.collision_count >= 3

        return state_vector, image, reward, done

    def _cleanup(self):
        """Destroy all actors."""
        actors = [self.camera, self.collision_sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.camera = None
        self.collision_sensor = None
        self.vehicle = None

    def close(self):
        """Clean up and restore async mode."""
        self._cleanup()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        print("Environment closed.")

    def get_weather_variants(self):
        """Return list of weather conditions for data collection."""
        return [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset,
        ]