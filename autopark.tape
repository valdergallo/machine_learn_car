# VHS Tape File for Recording Autonomous Parking Simulation to GIF
# To generate the GIF, run: vhs autopark.tape

Output autopark.gif

Set Shell "bash"
Set FontSize 14
Set Width 780
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"

# Run the simulation with smooth animation speed
Type "python3 autopark.py --speed 0.18"
Enter

# Let the simulation train and park the car
Sleep 6s
