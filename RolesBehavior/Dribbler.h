option(Dribbler)
{
	float passByKickX = theBehavior2015Parameters.passByKickX;
	float passByKickY = theBehavior2015Parameters.passByKickY;
	static Vector2f target(4500.f, 0.f);

	common_transition
	{

	}
	initial_state(start)
	{
		transition
		{
			goto dribble;
		}
		action
		{
			Stand();
		}
	}
	state(dribble)
	{
		transition
		{

		}
		action
		{
			switch(theBehaviorAction.behavior_action)
			{
				case 0:WalkAtSpeedPercentage(Pose2f(0.f, 0.2f, 0.f));break;
				case 1:WalkAtSpeedPercentage(Pose2f(0.f, 0.4f, 0.f));break;
				case 2:WalkAtSpeedPercentage(Pose2f(0.f, 0.6f, 0.f));break;
				case 3:WalkAtSpeedPercentage(Pose2f(0.f, 0.8f, 0.f));break;
				case 4:WalkAtSpeedPercentage(Pose2f(0.f, 1.0f, 0.f));break;
				case 5:Stand();break;
				case 6:AlignMent(Vector2f(4500.f,0.f),0,Vector2f(passByKickX,passByKickY));break;
				case 7:AlignMent(Vector2f(4500.f,0.f),0,Vector2f(passByKickX,passByKickY));break;
				case 8:AlignMent(Vector2f(-4500.f,0.f),0,Vector2f(passByKickX,passByKickY));break;
				case 9:WalkAtSpeedPercentage(Pose2f(0.f, 0.f, 1.f));break;
				default:Stand();break;
				std::cout << "action is ---> " << theBehaviorAction.behavior_action << std::endl;
				break;
			}

			theHeadControlMode = HeadControl::lookAtBall;
		}
	}
}
