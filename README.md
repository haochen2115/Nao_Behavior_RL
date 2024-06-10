# Nao_Behavior_RL
A Reinforcement Learning Approach to RoboCup Soccer-Robot Behavior Control

## Repository Contents
1. **GameController**
2. **BaseActionLearner**
3. **BehaviorLearner**
4. **RolesBehavior**
5. **ActionSet**

## Setup Instructions
To use these files based on the [Bhuman code release](https://github.com/bhuman/BHumanCodeRelease/releases/tag/coderelease2017), follow these steps:

1. Add `BehaviorAction` provider in `modules.cfg`.
2. Create a `resultPath` file in `Config/Locations/Default/`.
3. Add the following code to **/Make/Common/Nao.mare** and **/Make/Common/SimulatedNao.mare**:
    ```makefile
    "$(utilDirRoot)/tiny-dnn"
    ```
4. Add `tiny_dnn/` to `Util/`.
5. Replace `GameController.cpp` and `GameController.h`.
6. Add `REQUIRES(BehaviorAction)` to `BehaviorControlYEAR.h`.
7. Add the defined role behavior to `Options.h`.
8. Modify `PlayingState.h` and `RoleAssign.h`.
9. Add representation stream declaration.
10. Make other necessary changes as needed.
