option(RlearnerHighlevel)
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
				case 0:Kick(target);break;
				case 1:Dribble(target.x(), target.y());break;
				default:Stand();
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
