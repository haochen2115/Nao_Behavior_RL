/*
 * BehaviorLearnersaml.cpp
 * @haochen2115
 */

#include "BehaviorLearnersaml.h"

#include <iostream>
BehaviorLearnersaml::BehaviorLearnersaml() {
  if (theOwnTeamInfo.teamColor == TEAM_BLUE) {
    startTime = Time::getRealSystemTime();
    File pathFile("resultPath", "r");
    std::string filePath = pathFile.getFullName();
    std::ifstream inputStream(filePath);
    inputStream >> resultsFolder;
    #ifdef DQN
      initQNetwork();
    #endif
    #ifdef QTABLE
      initQTable();
    #endif
    resultsFolderAddr = "Time@" + std::to_string(startTime) + " | "  + "resultsFolderAddr is -> " + resultsFolder;
    std::cout << resultsFolderAddr << std::endl;
    std::cout << "initialized" << std::endl;
  }
}
void BehaviorLearnersaml::update(BehaviorAction& behaviorAction) {
  if (theOwnTeamInfo.teamColor == TEAM_RED) return;
  if (theGameInfo.secsRemaining > 0 && theGameInfo.secsRemaining < 600) {
    start = true;
  }
  if (!start) {
    return;
  }
  Vector2f ball = theBallModel.estimate.position;
  getState();//更新ballstatebin和opponentstatebin;
  getGameScene();//更新balliscontrolledbyopponent,intermediatereward;
  getGameState();//更新goalbyblue,goalbyred,ownScore,opponentScore,goalsScoredInRow,lastAttemptGoal,terminalreward

  static int goal100 = 0;
  static int stair = 0;
  if(goalbyblue||goalbyred)
  {
    if (TRAINING) {
        std::cout << "update after goal\n";
        if(hasreward_.size() > 1){
          #ifdef N_STEP_BACK_TD
            updateQTableAfterGoal(terminalreward);
          #endif
          #ifdef Q_LAMDA
            updateQTableAfterGoallamda(terminalreward);
          #endif
          #ifdef SARSA
            updateQTableAfterGoalsarsa(terminalreward);
          #endif
          #ifdef DQN
            putStateintoMemoryPoolfirsttime(terminalreward);
            updateQNetwork();
          #endif

          hasreward_[hasreward_.size()-2] = 1;
        }
        else
        {
          #ifdef N_STEP_BACK_TD
            updateQTableforFirstAction(terminalreward);
          #endif
          #ifdef Q_LAMDA
            updateQTableforFirstActionlamda(terminalreward);
          #endif
          #ifdef SARSA
            updateQTableforFirstActionsarsa(terminalreward);
          #endif
          #ifdef DQN
            putStateintoMemoryPoolfirsttime(terminalreward);
            updateQNetwork();
          #endif
        }
    }
    if(TRAINING)
    {
      #ifdef QTABLE
        writeQTableToFile();
      #endif
      #ifdef DQN
        writeToFile();
        writeLatestQNetworkToFile();
      #endif
    }
    if(amountTrials%100 == 0)
    {
      float rate = (float)goal100/100.f;
      write100rate(rate);
      goal100 = 0;
    }
    if(goalbyblue)
    {
      goal100++;
      writescoreepisode(amountTrials);
      stair++;

    }
    writescorestair(stair);
    if(fabs(epireward) > 100)
      epireward = -1;
    writeepisodereward(epireward);//this episode
    static int collectedreward = 0;
    collectedreward+=epireward;
    epireward = 0;
    writecollectedreward(collectedreward);//all episode
    amountTrials++;
    start = false;
    ball_states_.clear();
    opponent_states_.clear();
    actions_.clear();
    hasreward_.clear();
    qValues_.clear();
  }
  if(ball.norm() < CLOSE_DISTANCE && start) {
    if(ball_states_.size() == 0)
    {
    	ball_states_.push_back(ballstatebin-1);
    	opponent_states_.push_back(opponentstatebin-1);
    	hasreward_.push_back(0);
      #ifdef QTABLE
        selectActionFromeQTable();
      #endif
      #ifdef DQN
        selectActionFromQNetwork();
      #endif
      std::cout << "the first time to select actions\n";
    }
    else if(ballstatebin-1 != ball_states_.back() || opponentstatebin-1 != opponent_states_.back())
    {
    	ball_states_.push_back(ballstatebin-1);
    	opponent_states_.push_back(opponentstatebin-1);
    	hasreward_.push_back(0);
      #ifdef QTABLE
        selectActionFromeQTable();
      #endif
      #ifdef DQN
        selectActionFromQNetwork();
      #endif

    	if(TRAINING){
    		if(hasreward_[hasreward_.size()-2] == 0){
    			hasreward_[hasreward_.size()-2] = 1;
          #ifdef N_STEP_BACK_TD
             updateQTable(intermediatereward);
          #endif
          #ifdef Q_LAMDA
             updateQTablelamda(intermediatereward);
          #endif
          #ifdef SARSA
             updateQTablesarsa(intermediatereward);
          #endif
          #ifdef DQN
            putStateintoMemoryPool(intermediatereward);
            updateQNetwork();
          #endif
          std::cout << "state change\n";
    			
    		}
    	}
    }
  }
  else if(start)
  {
  	if(balliscontrolledbyopponent == true || ballisbehindme == true)
  	{
      if(hasreward_.size() > 0){
    		if(hasreward_[hasreward_.size()-1] == 0){
          std::cout << "controlled by opponent\n";
          std::cout << "punish last action\n";

          #ifdef N_STEP_BACK_TD
             updateQTable(intermediatereward);
          #endif
          #ifdef Q_LAMDA
             updateQTablelamda(intermediatereward);
          #endif
          #ifdef SARSA
             updateQTablesarsa(intermediatereward);
          #endif
          #ifdef DQN
            putStateintoMemoryPool(intermediatereward);
            updateQNetwork();
          #endif
    			hasreward_[hasreward_.size()-1] = 1;
        }
      }
  	}
  }
  if(actions_.size() > 0)
    behaviorAction.behavior_action = actions_.back();
 }
//离散化球的位置，传入的是全局的坐标
int BehaviorLearnersaml::discretizeBallPosition(Vector2f ballposition){
	int ballfieldx = ballposition.x();
	int ballfieldy = ballposition.y();
	int ballfieldstatebin = 1;
	if(ballfieldx<=0 || std::abs(ballfieldy) > 3000.f) 
		ballfieldstatebin = 1;
	else
		ballfieldstatebin = (ballfieldx/500 + 1) + ((3000 - ballfieldy)/1000)*3;
	return ballfieldstatebin;
}
//离散化对手的位置,传入的是相对机器人的坐标
int BehaviorLearnersaml::discretizeOpponentPosition(Vector2f opponentposition){
	Vector2f opponentfield = theRobotPose * opponentposition;
  int opponentfieldx = opponentfield.x();
  int opponentfieldy = opponentfield.y();
  int oprobotstate = 1;
  if(opponentfieldx<=0 || std::abs(opponentfieldy) > 3000.f) 
    oprobotstate = 1;
  else
    oprobotstate = (opponentfieldx/500 + 1) + ((3000 - opponentfieldy)/1000)*3;
  return oprobotstate;
}
//更新ballstatebin和opponentstatebin;记录xPositions_,yPositions_;
void BehaviorLearnersaml::getState(){
  int xPos = int(theRobotPose.translation.x());
  int yPos = int(theRobotPose.translation.y());
  if (xPositions_.size() == 0 || (xPos != xPositions_.back() || yPos != yPositions_.back())) {
    xPositions_.push_back(xPos);
    yPositions_.push_back(yPos);
  }
  Vector2f ball = theBallModel.estimate.position;
  Vector2f ballfield;
  ballfield = theRobotPose*ball;
  //先把球变成全局的再传进去
  ballstatebin = discretizeBallPosition(ballfield);
  //opponent用相对的就好了
  std::vector<Obstacle> obstacles = theObstacleModel.obstacles;
  for (Obstacle obstacle : obstacles) {
    if (obstacle.type == Obstacle::opponent) {
      opponentstatebin = discretizeOpponentPosition(obstacle.center);
      break;
    }
  }
}
//更新balliscontrolledbyopponent,intermediatereward;
void BehaviorLearnersaml::getGameScene(){
	Vector2f ball = theBallModel.estimate.position;
	Vector2f ballspeed = theBallModel.estimate.velocity;
	Vector2f op;
  std::vector<Obstacle> obstacles = theObstacleModel.obstacles;
	for (Obstacle obstacle : obstacles) {
	    if (obstacle.type == Obstacle::opponent) {
	      op = obstacle.center;
	      break;
	    }
  	} 
	bool ballisclosetoop = ((ball - op).norm() < 500.f);
	bool ballisfarfromme = (ball.norm() > 500.f);
	if(ballspeed.norm() < 50.f && ballisclosetoop && ballisfarfromme)
	{
		balliscontrolledbyopponent = true;
    ballisbehindme = false;
		intermediatereward = -2;
	}
	else if((theRobotPose * ball).x() < theRobotPose.translation.x() - 500.f)
	{
		balliscontrolledbyopponent = false;
    ballisbehindme = true;
		intermediatereward = -2;
	}
  else
  {
    balliscontrolledbyopponent = false;
    ballisbehindme = false;
    intermediatereward = 0;
  }
}
//更新goalbyblue,goalbyred,ownScore,opponentScore,goalsScoredInRow,lastAttemptGoal,terminalreward
void BehaviorLearnersaml::getGameState(){
  if (theOwnTeamInfo.score > ownScore) {//如果得分的话，奖赏为剩余时间奖励加2,嗯所以一旦进球，奖励就会大于2
    terminalreward = 10;
    ++ownScore;
    if (lastAttemptGoal) {
      ++goalsScoredInRow;
    }
    lastAttemptGoal = true;
    goalbyblue = true;
  } else if (theOpponentTeamInfo.score > opponentScore) {
    terminalreward = -1;
    std::cout << "scored by red\n";
    ++opponentScore;
    lastAttemptGoal = false;
    goalsScoredInRow = 0;
    goalbyred = true;
  }
  else
  {
  	goalbyblue = false;
  	goalbyred = false;
  }
  if (ownScore == 255 && theOwnTeamInfo.score == 0) {
    terminalreward = 10;
    ownScore = 0;
    if (lastAttemptGoal) {
      ++goalsScoredInRow;
    }
    lastAttemptGoal = true;
    // goalbyblue = true;
  }
  else if (opponentScore == 255 && theOpponentTeamInfo.score == 0) {
    terminalreward = -1;
    std::cout << "red scored over 255\n";
    opponentScore = 0;
    lastAttemptGoal = false;
    goalsScoredInRow = 0;
    // goalbyred = true;
  }
}
//写得分episode，xPositions_,yPositions_，qvalue，lastaction，和nn
void BehaviorLearnersaml::writeToFile(){
  std::cout << "start to write\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "q_neural_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::ofstream outputStream(modelFilePath);
  outputStream << nn;
  // write evaluation information to file
  std::cout << "write nn finish\n";
}
//把action以添加的方式写入文件
void BehaviorLearnersaml::writeActionToFile(){
  std::string actionFileName = resultsFolder + "action_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(actionFileName, "w");
  std::string actionFilePath = modelFile.getFullName();
  std::ofstream os(actionFilePath);
  os << "action --> refer to .csv file" << "\n";
  std::ofstream outFile;  
  outFile.open(actionFilePath + ".csv", std::ios::app);

  std::cout << "actions_ ---> " << actions_.size() << std::endl;
  if(actions_.size() > 0){
    for (int action : actions_) {
      outFile << action << ", ";
    }
    outFile << "\n";
  }
  outFile.close(); 
}

void BehaviorLearnersaml::write100rate(float rate){
  std::string rate100filename = resultsFolder + "middle_rate100.csv";
  std::ofstream outFile;  
  outFile.open(rate100filename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << rate << ",";
  outFile.close();
}
void BehaviorLearnersaml::writescoreepisode(int scoreepisode){
  std::string scoreepisodefilename = resultsFolder + "middle_scoreepisode.csv";
  std::ofstream outFile;  
  outFile.open(scoreepisodefilename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << scoreepisode << ",";
  outFile.close();
}
void BehaviorLearnersaml::writescorestair(int scorestair){
  std::string scorestairfilename = resultsFolder + "middle_scorestair.csv";
  std::ofstream outFile;  
  outFile.open(scorestairfilename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << scorestair << ",";
  outFile.close();
}
void BehaviorLearnersaml::writeepisodereward(int episodereward){
  std::string episoderewardfilename = resultsFolder + "middle_episodereward.csv";
  std::ofstream outFile;  
  outFile.open(episoderewardfilename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << episodereward << ",";
  outFile.close();
}
void BehaviorLearnersaml::writecollectedreward(int collectedreward){
  std::string collectedrewardfilename = resultsFolder + "middle_collectedreward.csv";
  std::ofstream outFile;  
  outFile.open(collectedrewardfilename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << collectedreward << ",";
  outFile.close();
}

void BehaviorLearnersaml::initQTable(){
  if (TRAINING && !LOAD_FROM_FILE) {
  } else {
    // Read network from file
    std::string fileName = resultsFolder + "q_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
    std::cout << "Loaded from " << fileName << std::endl;
    File modelFile(fileName, "r");
    std::string modelFilePath = modelFile.getFullName();
    std::cout << "Loaded from " << modelFilePath << std::endl;
    std::ifstream inputStream(modelFilePath);
    for (int i = 0; i < AMOUNT_BALL_STATES*AMOUNT_OPPONENT_STATES; ++i) {
      for (int j = 0; j < AMOUNT_ACTIONS; ++j) {
        inputStream >> QTable[i][j];
      }
    }
  }
  std::cout << std::endl;
}
void BehaviorLearnersaml::updateQTable(float reward){
  epireward+=reward;
  std::cout << "updating QTable using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = 1;
    for (int i = ball_states_.size() - 2; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += (reward * updateWeight);
      std::cout << "add ---> " << (reward * updateWeight);
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= 0.7;
    }
  }
}
void BehaviorLearnersaml::updateQTableAfterGoal(float reward){
  epireward+=reward;
  std::cout << "updating QTable after goal using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = 1;
    for (int i = ball_states_.size() - 1; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += (reward * updateWeight);
      std::cout << "add ---> " << (reward * updateWeight);
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= 0.7;
    }
  }
}
void BehaviorLearnersaml::updateQTableforFirstAction(float reward){
  epireward+=reward;
  // std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
  if (actions_.size() == 1) {
    std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
    int ballstate = ball_states_[ball_states_.size() - 1];
    int opponentstate = opponent_states_[opponent_states_.size() - 1];
    int action = actions_[actions_.size() - 1];
    QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] += reward;
  }
}
void BehaviorLearnersaml::updateQTablelamda(float reward){
  epireward+=reward;
  std::cout << "updating lambda QTable using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = LAMDA;
      output = qValues_[ ball_states_.size() - 2 ];
    float delta = reward + GAMMA* (*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) - output[actions_[actions_.size() - 2]];
    for (int i = ball_states_.size() - 2; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      std::cout << "add ---> " << ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersaml::updateQTableAfterGoallamda(float reward){
  epireward+=reward;
  std::cout << "updating lambda QTable after goal using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = LAMDA;
    output = qValues_[ ball_states_.size() - 1 ];
    float delta = reward + GAMMA* (*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) - output[actions_[actions_.size() - 2]];
    for (int i = ball_states_.size() - 1; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      std::cout << "add ---> " << ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersaml::updateQTableforFirstActionlamda(float reward){
  epireward+=reward;
  // std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
  if (actions_.size() == 1) {
    std::cout << "updating lambda QTable for 1st action using reward ---> " << reward << std::endl;
    int ballstate = ball_states_[ball_states_.size() - 1];
    int opponentstate = opponent_states_[opponent_states_.size() - 1];
    int action = actions_[actions_.size() - 1];
    QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] += ALPHA * reward;
  }
}

void BehaviorLearnersaml::updateQTablesarsa(float reward){
  epireward+=reward;
  std::cout << "updating sarsa QTable using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = LAMDA;
    tiny_dnn::vec_t sarsa_next;
    sarsa_next = qValues_[ ball_states_.size() - 1 ];
    output = qValues_[ ball_states_.size() - 2 ];
    float delta = reward + GAMMA * (sarsa_next[actions_[actions_.size() - 1]]) - output[actions_[actions_.size() - 2]];
    for (int i = ball_states_.size() - 2; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      std::cout << "add ---> " << ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersaml::updateQTableAfterGoalsarsa(float reward){
  epireward+=reward;
  std::cout << "updating sarsa QTable after goal using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = LAMDA;
    output = qValues_[ ball_states_.size() - 1 ];
    float delta = reward;
    for (int i = ball_states_.size() - 1; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      std::cout << "add ---> " << ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersaml::updateQTableforFirstActionsarsa(float reward){
  epireward+=reward;
  // std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
  if (actions_.size() == 1) {
    std::cout << "updating sarsa QTable for 1st action using reward ---> " << reward << std::endl;
    int ballstate = ball_states_[ball_states_.size() - 1];
    int opponentstate = opponent_states_[opponent_states_.size() - 1];
    int action = actions_[actions_.size() - 1];
    QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] += ALPHA * reward;
  }
}

void BehaviorLearnersaml::selectActionFromeQTable(){
  int stateVector[AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES] = {0};
  stateVector[ball_states_.back()] = 1;
  stateVector[AMOUNT_BALL_STATES + opponent_states_.back()] = 1;
  // Predict qValues
  int ballstate = ball_states_.back();
  int opponentstate = opponent_states_.back();

  // tiny_dnn::vec_t qValues = nn.predict(stateVector);
  tiny_dnn::vec_t qValues;
  for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
    qValues.push_back(QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][i]);
  }
  int bestAction = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
  if (!TRAINING) {
    actions_.push_back(bestAction);
    return;
  }
  //Boltzmann action selection
  double maxQValue = double(*std::max_element(qValues.begin(), qValues.end()));
  double sumProbabilities = 0;
  std::vector<double> probabilities(AMOUNT_ACTIONS);
  for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
    // probabilities[i] = exp((qValues[i] - maxQValue) / TAU);
    if(fabs(maxQValue) > 1.f)
      probabilities[i] = exp((double)(qValues[i] - maxQValue) / fabs(maxQValue) * TAU);
    else
      probabilities[i] = exp((qValues[i]) / TAU);
    sumProbabilities += probabilities[i];
    std::cout << "qvalue: " << qValues[i] << std::endl;
    // std::cout << "e: " << probabilities[i] << std::endl;    
  }
  // Get random action from probability distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(20,80);
  // std::discrete_distribution<> d(probabilities.begin(), probabilities.end());
  std::uniform_int_distribution<> dis(0, AMOUNT_ACTIONS-1);
  int random_action = dis(gen);
  if(d(gen) == 0){
    std::cout << "exploration" << std::endl;
    actions_.push_back(random_action);
  }
  else{
    std::cout << "use" << std::endl;
    if(qValues[bestAction] == 0){//如果是0,随机选择一个，增大探索空间～
      // int Time = Time::getRealSystemTime();
      // actions_.push_back(Time%AMOUNT_ACTIONS);
      actions_.push_back(random_action);
    }
    else
      actions_.push_back(bestAction);
  }
  // actions_.push_back(d(gen));
  qValues_.push_back(qValues);
  std::cout << "BehaviorLearner Select Actions" << std::endl;
  std::cout << " you should take (" << actions_.back() << ")th action" << std::endl;

}
void BehaviorLearnersaml::writeQTableToFile(){
  std::cout << "start to write\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "q_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::ofstream outputStream(modelFilePath);
  for (int i = 0; i < AMOUNT_BALL_STATES*AMOUNT_OPPONENT_STATES; ++i) {
    for (int j = 0; j < AMOUNT_ACTIONS; ++j) {
      outputStream << QTable[i][j] << " ";
    }
  }
}
void BehaviorLearnersaml::writeScoreAndPosition(){
  int elapsedTime = Time::getRealTimeSince(startTime);
  std::cout << "#" << amountTrials << " amount of trials." << std::endl;
  std::cout << "This took " << elapsedTime/1000 << "seconds (" << (elapsedTime/1000)/60 << "minutes)" << std::endl;
  std::string evaFileName = resultsFolder + "evaluation_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES) + "_" + std::to_string(amountTrials);
  File evaluationFile(evaFileName, "w");
  std::string evaluationFilePath = evaluationFile.getFullName();
  std::ofstream os(evaluationFilePath);
  os << amountTrials << " " << elapsedTime/1000 << "\n";
  // os << amountTrials << "\n";
  os << "scored in episodes: \n [";
  for (int episode : scoredInEpisodes) {
    os << episode << ", ";
  }
  os << "] \n xPositions: \n [";
  for (int x : xPositions_) {
    os << x << ", ";
  }
  os << "] \n yPositions: \n [";
  for (int y : yPositions_) {
    os << y << ", ";
  }
  os << "] \n";
  if(actions_.size() > 0)
    os << "last action: " << actions_.back();

  os << "\n last qValues: \n";
  if(qValues_.size() > 0){
    tiny_dnn::vec_t lastQ = qValues_.back();
    for (int i = 0; i < int(lastQ.size()); ++i) {
      os << i << ": " << lastQ[i] << "\n";
    }
  }
  //for convenience of operating with python
  std::ofstream outFile;  
  outFile.open(evaluationFilePath + ".csv", std::ios::out);  
  outFile<<"xPositions"<<",";
  for (int x : xPositions_) {
    outFile << x << ", ";
  }
  outFile << std::endl;
  outFile<<"yPositions"<<",";
  for (int y : yPositions_) {
    outFile << y << ", ";
  }
  outFile << std::endl;
  outFile.close(); 
  xPositions_.clear();
  yPositions_.clear();
  std::cout << "write to file function finish\n";
}
void BehaviorLearnersaml::initQNetwork(){
  nn << tiny_dnn::fc<tiny_dnn::activation::relu>(QUANTITY_STATES, QUANTITY_ACTIONS)//2还是54+54；
     << tiny_dnn::fc<tiny_dnn::activation::relu>(QUANTITY_ACTIONS, QUANTITY_ACTIONS)
     << tiny_dnn::fc<tiny_dnn::activation::relu>(QUANTITY_ACTIONS, QUANTITY_ACTIONS)
     << tiny_dnn::fc<tiny_dnn::activation::identity>(QUANTITY_ACTIONS, QUANTITY_ACTIONS);
  if (TRAINING && !LOAD_FROM_FILE) {
    // Initialize weights and bias
    nn.weight_init(tiny_dnn::weight_init::xavier(1.0));
    nn.bias_init(tiny_dnn::weight_init::constant(0.0));
    nn.init_weight();
  } else {
    // Read network from file
    std::string fileName = "q_neural_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
    std::cout << "Loaded from " << fileName << std::endl;
    File modelFile(fileName, "r");
    std::string modelFilePath = modelFile.getFullName();
    std::cout << "Loaded from " << modelFilePath << std::endl;
    std::ifstream inputStream(modelFilePath);
    inputStream >> nn;
  }
  std::cout << std::endl;
  nn_update = nn;
}
void BehaviorLearnersaml::putStateintoMemoryPool(float reward){//选完动作准备更新的时候调用一下；
  // if(ball_states_.size() > 1) 第一次不调用这个，只在执行了一次之后才压入记忆池～
  if(memory_ball_states_.size() < MEMORY_POOL_CAPACITY)
  {
    position = 0;

    tiny_dnn::vec_t stateVector;
    stateVector.push_back(ball_states_[ball_states_.size() - 2]);
    stateVector.push_back(opponent_states_[opponent_states_.size() - 2]);
    memory_input.push_back(stateVector);

    tiny_dnn::vec_t output;
    output = qValues_[qValues_.size() - 2];
    float delta = reward + GAMMA*(*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) -  output[actions_[actions_.size() - 2]];//Q
    output[actions_[actions_.size() - 2]] += ALPHA*delta;
    memory_qValues.push_back(output);
  }
  else
  {
    tiny_dnn::vec_t stateVector;
    stateVector.push_back(ball_states_[ball_states_.size() - 2]);
    stateVector.push_back(opponent_states_[opponent_states_.size() - 2]);

    memory_input[position] = stateVector;

    tiny_dnn::vec_t output;
    output = qValues_[qValues_.size() - 2];
    float delta = reward + GAMMA*(*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) -  output[actions_[actions_.size() - 2]];//Q
    output[actions_[actions_.size() - 2]] += ALPHA*delta;

    memory_qValues[position] = output;

    position++;
    if(position == MEMORY_POOL_CAPACITY) position = 0;
  }
}
void BehaviorLearnersaml::putStateintoMemoryPoolfirsttime(float reward){
  if(memory_ball_states_.size() < MEMORY_POOL_CAPACITY)
  {
    position = 0;

    tiny_dnn::vec_t stateVector;
    stateVector.push_back(ball_states_[ball_states_.size() - 1]);
    stateVector.push_back(opponent_states_[opponent_states_.size() - 1]);
    memory_input.push_back(stateVector);

    tiny_dnn::vec_t output;
    output = qValues_[qValues_.size() - 1];
    float delta = reward;//Q
    output[actions_[actions_.size() - 1]] += ALPHA*delta;
    memory_qValues.push_back(output);
  }
  else
  {
    tiny_dnn::vec_t stateVector;
    stateVector.push_back(ball_states_[ball_states_.size() - 1]);
    stateVector.push_back(opponent_states_[opponent_states_.size() - 1]);

    memory_input[position] = stateVector;

    tiny_dnn::vec_t output;
    output = qValues_[qValues_.size() - 1];
    float delta = reward;//Q
    output[actions_[actions_.size() - 1]] += ALPHA*delta;

    memory_qValues[position] = output;

    position++;
    if(position == MEMORY_POOL_CAPACITY) position = 0;
  }
}
//更新qnn,memory_input 和 memory_qValues 需要提前压入记忆池
void BehaviorLearnersaml::updateQNetwork(){
  //如果小于miniBatch，不更新，否则shuffle更新，
  //记录次数，每C次更新，替换一次nn
  if(memory_num.size() < miniBatch)
    return;
  else{
    //shuffle
    sample_num = memory_num;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sample_num.begin(), sample_num.end(), g);
    //相应的sar压入input,qValue;
    std::vector<tiny_dnn::vec_t> input;
    std::vector<tiny_dnn::vec_t> qValues;
    for(int i = 0 ; i < miniBatch ; i++){
      input.push_back(memory_input[sample_num[i]]);
      qValues.push_back(memory_qValues[sample_num[i]]);
    }
    //update
    tiny_dnn::gradient_descent opt;//梯度下降
    nn_update.fit<tiny_dnn::mse>(opt, input, qValues, BATCH_SIZE, EPOCHS);//BATCH_SIZE 1/50 ~

    //计数，每C次替换
    update_count++;
    if(update_count == C_STEP_CHANGE){
      std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!change network" << std::endl;
      nn = nn_update;
      update_count = 0;
    }
  }
} 
//使用ddqn时需要用到
void BehaviorLearnersaml::writeLatestQNetworkToFile(){
  std::cout << "start to write q neural latest version\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "q_neural_latest_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::ofstream outputStream(modelFilePath);
  outputStream << nn_update;
  // write evaluation information to file
  std::cout << "write nn_update finish\n";
}

void BehaviorLearnersaml::selectActionFromQNetwork(){
  int stateVector[QUANTITY_STATES] = {0};
  stateVector[0] = ball_states_.back();
  stateVector[1] = opponent_states_.back();
  // Predict qValues
  tiny_dnn::vec_t qValues = nn.predict(stateVector);
  int bestAction = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
  if (!TRAINING) {
    actions_.push_back(bestAction);
    return;
  }
  //Boltzmann action selection
  double maxQValue = double(*std::max_element(qValues.begin(), qValues.end()));
  double sumProbabilities = 0;
  std::vector<double> probabilities(QUANTITY_ACTIONS);
  for (int i = 0; i < QUANTITY_ACTIONS; ++i) {
    // probabilities[i] = exp((qValues[i] - maxQValue) / TAU);
    if(fabs(maxQValue) > 1.f)
      probabilities[i] = exp((double)(qValues[i] - maxQValue) / fabs(maxQValue) * TAU);
    else
      probabilities[i] = exp((qValues[i]) / TAU);
    sumProbabilities += probabilities[i];
    std::cout << "qvalue: " << qValues[i] << std::endl;
    // std::cout << "e: " << probabilities[i] << std::endl;    
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(20,80);
  // std::discrete_distribution<> d(probabilities.begin(), probabilities.end());
  std::uniform_int_distribution<> dis(0, QUANTITY_ACTIONS-1);
  int random_action = dis(gen);
  if(d(gen) == 0){
    std::cout << "exploration" << std::endl;
    actions_.push_back(random_action);
  }
  else{
    std::cout << "use" << std::endl;
    actions_.push_back(bestAction);
  }
  // actions_.push_back(d(gen));
  qValues_.push_back(qValues);
  std::cout << "BehaviorLearner Select Actions" << std::endl;
  std::cout << " you should take (" << actions_.back() << ")th action" << std::endl;

}

MAKE_MODULE(BehaviorLearnersaml, behaviorControl)
