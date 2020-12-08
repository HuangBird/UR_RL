# UR5_RL_Assembly
## Description
Use RL algorithm to help the UR5 robot learn to finish an assembly task (peg in hole).

## Environment
- tensorflow-gpu 1.14.0

## V-REP
Install V-REP, see the website at http://www.coppeliarobotics.com/downloads.html for installation instructions.
Make sure you have following files in your directory, in order to run the various examples:
1. vrep.py
2. vrepConst.py
3. the appropriate remote API library: "remoteApi.dll" (Windows), "remoteApi.dylib" (Mac) or "remoteApi.so" (Linux)

## Steps
- Run V-REP:
    ```
    cd CoppeliaSim_Edu_V4_0_0_Ubuntu16_04
    ./coppeliaSim.sh
    ```

- Open `ur5_force_vision_insertion.ttt` in V-REP. 
- Run `main.py`.
