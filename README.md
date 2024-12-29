# sensor_perception

This repository contains a LIDAR-based robotic sensing model that leverages the Doppler effect to estimate the relative velocities of moving objects in a dynamic 2D environment. By integrating ray tracing for distance measurements with Doppler-induced frequency shifts, the code simulates a more realistic sensor that can detect both range and velocity. Gaussian noise is added to mimic real-world conditions.

Two main obstacle-avoidance strategies are demonstrated:

1. Basic Start-Stop: The robot drives straight toward the goal, pausing when an obstacle is detected and resuming once it’s clear.
2. A* Search Algorithm: Dynamically plans the path by predicting obstacle positions based on Doppler-enhanced velocity estimates, continuously updating the route to avoid collisions.

   
Key Features

1. Ray Tracing for LIDAR: Computes intersection points between laser beams and circular obstacles, accounting for round-trip delay times.
2. Doppler Velocity Estimation: Calculates velocity along the line of sight by detecting frequency shifts in the returned laser pulses.
3. Gaussian Noise Model: Introduces random variations into Doppler measurements to simulate sensor imperfections.
4. Robot/Obstacle Motion: Incorporates translational and rotational dynamics for both the robot and moving obstacles.
5. Start-Stop vs. A*: Compares simple linear navigation with a graph-based path planner that uses Doppler data to improve collision avoidance.

Use this code to explore how Doppler-enhanced measurements can improve perception and path planning in autonomous robotic applications.
