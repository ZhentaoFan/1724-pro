import json
import matplotlib.pyplot as plt
import argparse


# with open("../data/Hangzhou/4_4/roadnet_4_4.json", 'r') as road_file:
#     road_data = json.load(road_file)

# # Load the newly uploaded data
# with open("../data/Hangzhou/4_4/anon_4_4_hangzhou_real.json", "r") as file:
#     new_volume_data = json.load(file)
    
with open("../data/Jinan/3_4/roadnet_3_4.json", 'r') as road_file:
    road_data = json.load(road_file)

# Load the newly uploaded data
with open("../data/Jinan/3_4/anon_3_4_jinan_real.json", "r") as file:
    new_volume_data = json.load(file)
    
# Calculate the traffic volume on each road segment

# Initialize a dictionary to store the traffic volume for each road segment
traffic_volume = {}

# Loop through each vehicle's route to compute the traffic volume
for entry in new_volume_data:
    route = entry["route"]
    for road_segment in route:
        if road_segment not in traffic_volume:
            traffic_volume[road_segment] = 0
        traffic_volume[road_segment] += 1

# Visualize the (3|4)x4 road map with roads colorized based on traffic volume

fig, ax = plt.subplots(figsize=(15, 15))

# Set a colormap for traffic volume
norm = plt.Normalize(min(traffic_volume.values()), max(traffic_volume.values()))
cmap = plt.get_cmap("Reds")

# Draw roads
for road in road_data["roads"]:
    start_point = road["points"][0]
    end_point = road["points"][1]
    road_id = road["id"]
    if road_id in traffic_volume:
        color = cmap(norm(traffic_volume[road_id]))
    else:
        color = "gray"  # Default color for roads with no traffic data
    plt.plot([start_point["x"], end_point["x"]], [start_point["y"], end_point["y"]], color=color, linewidth=2.5)

# Draw intersections
for intersection in road_data["intersections"]:
    plt.scatter(intersection["point"]["x"], intersection["point"]["y"], s=100, c='g', marker='o')

# Add colorbar to indicate traffic volume
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Traffic Volume")

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Road Network with Traffic Volume Colorized')
plt.grid(True)
plt.show()
