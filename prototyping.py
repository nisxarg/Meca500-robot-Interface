import mecademicpy.robot as mdr
import time

robot : mdr.Robot = mdr.Robot()
robot.Connect(address='192.168.0.100', enable_synchronous_mode=False)
robot.ActivateAndHome()
robot.WaitHomed()

robot.MoveJoints(0, 0, 0, 0, 0, 0)
robot.MoveJoints(0, -60, 60, 0, 0, 0)
print()

# Print robot position while it's moving
try:
    for _ in range(100):
        print(robot.GetJoints())
        print(robot.GetRobotRtData().rt_external_tool_status)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("ahh")
robot.MoveJoints(0, 0, 0, 0, 0, 0)

robot.WaitIdle()
robot.DeactivateRobot()
robot.WaitDeactivated()
robot.Disconnect()