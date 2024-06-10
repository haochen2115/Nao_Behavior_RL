option(RlearnerMiddlelevel)
{

	initial_state(start)
	{
		transition
		{
			goto trying;
		}
		action
		{
			Stand();
		}
	}
	state(trying)
	{
		transition
		{
			if(ball.positionRobot.norm() < CLOSE_DISTANCE)
			{
				goto selectaction;
			}
			else
				goto walktoball;
		}
		action
		{
			Stand();
		}
	}
	state(selectaction)
	{
		transition
		{
			if(ball.positionRobot.norm() > CLOSE_DISTANCE)
			{
				goto trying;
			}
		}
		action
		{
			switch(theBehaviorAction.behavior_action)
			{
				case 0:KickTo(0);break;//0_deg
				case 1:KickTo(0.524f);break;//30_deg
				case 2:KickTo(0.873f);break;//50_deg
				case 3:KickTo(1.222f);break;//70_deg
				case 4:KickTo(1.484f);break;//85_deg
				case 5:KickTo(-0.524f);break;//-30_deg
				case 6:KickTo(-0.873f);break;//-50_deg
				case 7:KickTo(-1.222f);break;//-70_deg
				case 8:KickTo(-1.484f);break;//-85_deg

				case 9:DribbleTo(0);break;//0_deg
				case 10:DribbleTo(0.785f);break;//45_deg
				case 11:DribbleTo(1.571f);break;//90_deg
				case 12:DribbleTo(2.356f);break;//135_deg
				case 13:DribbleTo(3.054f);break;//175_deg
				case 14:DribbleTo(-0.785f);break;//-45_deg
				case 15:DribbleTo(-1.571f);break;//-90_deg
				case 16:DribbleTo(-2.356f);break;//-135_deg

				default:Stand();break;
				break;
			}
		}
	}
	state(walktoball)
	{
		transition
		{
			if(ball.positionRobot.norm() < CLOSE_DISTANCE)
			{
				goto trying;
			}
		}
		action
		{
			WalkToBall();
		}
	}
}
