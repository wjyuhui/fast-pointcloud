# Single semantic target lifted to 3D (base_link).
std_msgs/Header header

int32 class_id
string class_name
float32 score

# 2D box on color image (xyxy pixels)
float32 x1
float32 y1
float32 x2
float32 y2

# Center in camera optical frame (x right, y down, z forward)
geometry_msgs/Point center_cam

# Center in base_link (x forward, y left, z up)
geometry_msgs/Point center_base

# Depth along camera optical Z used for lift (meters)
float32 distance_m
int32 num_valid_depth
