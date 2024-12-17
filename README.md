# osu! Tablet Area Optimizer (WIP)

A program to analyze hit errors in a map and provide recommended tilt and size adjustments.

![Built in GUI](./gui.png)

# How it works:

For each object in a map, an algorithm is used to determine the attempt made by the player to hit the object and a correspondance is built accordingly. We take each pair in this correspondance and find the signed angle between the vector from the center of the playfield to the object, and the vector from the center of the playfield to the hit attempt, and find the average error. For size adjustments, we simply find the mean of the ratio of the magnitude of these two vectors.

Size and tilt adjustments are then returned to the user, and the modifications plotted for easier visualization.

This project is still a work in progress. Future features could include a least-squared-error fit with tilt/size as parameters rather than a simple mean and an improved GUI to automate map selection given a home osu directory.
