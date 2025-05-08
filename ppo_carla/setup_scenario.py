import carla
import random
import time

def setup_carla_environment(town_name="Town01", num_vehicles=30, num_pedestrians=50, weather_id=0, seed=42):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.load_world(town_name)
    print(f"Loaded {town_name}")

    # Enable synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    random.seed(seed)

    weather_presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.WetCloudySunset
    ]
    world.set_weather(weather_presets[weather_id % len(weather_presets)])
    print(f"Weather set to preset {weather_id}")

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    for i in range(min(num_vehicles, len(spawn_points))):
        blueprint = random.choice(vehicle_blueprints)
        transform = spawn_points[i]
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle:
            vehicle.set_autopilot(True)

    walker_bp_library = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find("controller.ai.walker")

    walker_ids = []
    controller_ids = []
    walker_actors = []
    controller_actors = []

    spawn_locations = []
    for _ in range(num_pedestrians):
        location = world.get_random_location_from_navigation()
        if location:
            spawn_locations.append(carla.Transform(location))

    walker_batch = []
    for transform in spawn_locations:
        walker_bp = random.choice(walker_bp_library)
        walker_batch.append(carla.command.SpawnActor(walker_bp, transform))

    walker_results = client.apply_batch_sync(walker_batch, True)
    for result in walker_results:
        if not result.error:
            walker_ids.append(result.actor_id)

    controller_batch = []
    for walker_id in walker_ids:
        controller_batch.append(
            carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
        )

    controller_results = client.apply_batch_sync(controller_batch, True)
    for result in controller_results:
        if not result.error:
            controller_ids.append(result.actor_id)

    walker_actors = [world.get_actor(wid) for wid in walker_ids]
    controller_actors = [world.get_actor(cid) for cid in controller_ids]

    print("Starting controllers and assigning destinations...")
    for walker, controller in zip(walker_actors, controller_actors):
        if not walker or not controller:
            continue
        try:
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination:
                controller.go_to_location(destination)
                controller.set_max_speed(1.0 + random.random())
                print(f"Walker {walker.id} moving to {destination}")
        except Exception as e:
            print(f"Failed to control walker {walker.id}: {e}")

    print("Ticking the world to simulate movement...")
    for _ in range(200):  # Simulate ~10 seconds
        world.tick()
        time.sleep(0.05)

    print(f"Spawned {num_vehicles} vehicles and {num_pedestrians} pedestrians in {town_name}.")

    # Reset to async mode
    settings.synchronous_mode = False
    world.apply_settings(settings)

    return world

if __name__ == "__main__":
    setup_carla_environment(
        town_name="Town01",
        num_vehicles=40,
        num_pedestrians=60,
        weather_id=2,
        seed=2025
    )
