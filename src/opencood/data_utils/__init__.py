
SUPER_CLASS_MAP = {
    "vehicle": ["LongVehicle", "Car", "PoliceCar"],
    "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter",
                   "ScooterRider", "Motorcycle", "MotorcyleRider",
                   "BicycleRider"],
    "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"],
}
# Car, [Pedestrian, Scooter, Motorcycle, Bicycle], [Truck, Van, TrashCan, ConcreteTruck, Bus]

# SUPER_CLASS_MAP = {
#         "car": ["Car", "PoliceCar"],  # Most common class
#         "pedestrian": ["Child","RoadWorker","Pedestrian"],  # Second most common
#         "scooter": ["Scooter","ScooterRider"],  # Third most common
#         "motorcycle": ["Motorcycle","MotorcyleRider"],
#         "bicycle": ["BicycleRider"],
#         "truck": ["Truck","ConcreteTruck"],
#         "van": ["Van"],
#         "barrier": ["TrashCan"],  # Static obstacles
#         "box_truck": ["LongVehicle"],
#         "bus": ["Bus"]  # Largest vehicle class
#     }