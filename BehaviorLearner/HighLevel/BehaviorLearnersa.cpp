/*
 * BehaviorLearnersa.cpp
 * @haochen2115
 */

#include "BehaviorLearnersa.h"
#include <iostream>
BehaviorLearnersa::BehaviorLearnersa() {
  if (theOwnTeamInfo.teamColor == TEAM_BLUE) {
    startTime = Time::getRealSystemTime();
    File pathFile("resultPath", "r");
    std::string filePath = pathFile.getFullName();
    std::ifstream inputStream(filePath);
    inputStream >> resultsFolder;
    initQTable();
    amountTrials = CONTINUE_TRAIL_NUMBER;
    resultsFolderAddr = "Time@" + std::to_string(startTime) + " | "  + "resultsFolderAddr is -> " + resultsFolder;
    std::cout << resultsFolderAddr << std::endl;
    std::cout << "initialized" << std::endl;
  }
}
void BehaviorLearnersa::update(BehaviorAction& behaviorAction) {
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

          hasreward_[hasreward_.size()-2] = 1;
        //   // std::cout << "update after goal success\n";
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
        }
    }
    if(TRAINING)
    {
        writeQTableToFile();
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
    std::cout << "# " << amountTrials << " Trials\n";
    amountTrials++;
    start = false;
    ball_states_.clear();
    opponent_states_.clear();
    actions_.clear();
    hasreward_.clear();
    qValues_.clear();
  }  
  if(ball.norm() < CLOSE_DISTANCE) {
    if(ball_states_.size() == 0)
    {
    	ball_states_.push_back(ballstatebin-1);//把 bin-1 push_back因为下标是从0开始的
    	opponent_states_.push_back(opponentstatebin-1);
      selectActionFromeQTable();
    	hasreward_.push_back(0);
      std::cout << "the first time to select actions\n";
    }
    else if(ballstatebin-1 != ball_states_.back() || opponentstatebin-1 != opponent_states_.back())
    {
    	ball_states_.push_back(ballstatebin-1);
    	opponent_states_.push_back(opponentstatebin-1);
      selectActionFromeQTable();
    	hasreward_.push_back(0);
    	if(TRAINING){
    		if(hasreward_[hasreward_.size()-2] == 0){
          #ifdef N_STEP_BACK_TD
             updateQTable(intermediatereward);
          #endif
          #ifdef Q_LAMDA
             updateQTablelamda(intermediatereward);
          #endif
          #ifdef SARSA
             updateQTablesarsa(intermediatereward);
          #endif
    			// updateNetwork(intermediatereward);
          std::cout << "state change\n";
    			hasreward_[hasreward_.size()-2] = 1;
    		}
    	}
    }
  }
  else
  {
    // std::cout << "far from ball or passby\n";
  	if(balliscontrolledbyopponent == true)
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
    			hasreward_[hasreward_.size()-1] = 1;
        }
      }
  	}
  }
  if(actions_.size() > 0)
    behaviorAction.behavior_action = actions_.back();
}
//初始化网络
void BehaviorLearnersa::initNetwork(){
  nn << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::relu>(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES)
     << tiny_dnn::fc<tiny_dnn::activation::identity>(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, AMOUNT_ACTIONS);
  if (TRAINING && !LOAD_FROM_FILE) {
    // Initialize weights and bias
    nn.weight_init(tiny_dnn::weight_init::xavier(1.0));
    nn.bias_init(tiny_dnn::weight_init::constant(0.0));
    nn.init_weight();
  } else {
    // Read network from file
    std::string fileName = "neural_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
    std::cout << "Loaded from " << fileName << std::endl;
    File modelFile(fileName, "r");
    std::string modelFilePath = modelFile.getFullName();
    std::cout << "Loaded from " << modelFilePath << std::endl;
    std::ifstream inputStream(modelFilePath);
    inputStream >> nn;
  }
  std::cout << std::endl;
}
//离散化球的位置，传入的是全局的坐标
int BehaviorLearnersa::discretizeBallPosition(Vector2f ballposition){
	int ballfieldx = ballposition.x();
	int ballfieldy = ballposition.y();
	int ballfieldstatebin = 1;
	if(ballfieldx<=0 || std::abs(ballfieldy) > 3000.f) 
		ballfieldstatebin = 1;
	else 
		ballfieldstatebin = (ballfieldx/1500 + 1) + ((3000 - ballfieldy)/2000)*3;
	return ballfieldstatebin;
}
//离散化对手的位置,传入的是相对机器人的坐标
int BehaviorLearnersa::discretizeOpponentPosition(Vector2f opponentposition){
	int oprobotstate = 1;
	float angle = float(atan2(opponentposition.y(), opponentposition.x()));
  	int angleBin = int(round((angle / ( 2 * M_PI)) * 4)) + 2;
  	if (angleBin == 0) angleBin = 4;
	if(opponentposition.norm() < OPPONENT_FIRST_LEVEL_DISTANCE)
		oprobotstate = 1;
	else if(opponentposition.norm() < OPPONENT_SECOND_LEVEL_DISTANCE)
		oprobotstate = 1+angleBin;
	else
		oprobotstate = 5+angleBin;
	return oprobotstate;
}
//更新ballstatebin和opponentstatebin;记录xPositions_,yPositions_;
void BehaviorLearnersa::getState(){
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
      // needpassby = obstacle.center.norm() < CLOSE_DISTANCE && obstacle.center.x() > 10.f;
      break;
    }
  }  
}
//更新balliscontrolledbyopponent,intermediatereward;
void BehaviorLearnersa::getGameScene(){
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
		intermediatereward = -2;
	}
	else
	{
		balliscontrolledbyopponent = false;
		intermediatereward = 0;
	}
}
//更新goalbyblue,goalbyred,ownScore,opponentScore,goalsScoredInRow,lastAttemptGoal,terminalreward
void BehaviorLearnersa::getGameState(){
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

  // Set score back to 0 when 255 because unsigned char
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
//更新nn
void BehaviorLearnersa::updateNetwork(float reward){
  std::cout << "updating network using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    // Create input
    tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
    stateVector[ball_states_[ball_states_.size() - 2]] = 1; 
    stateVector[AMOUNT_BALL_STATES + opponent_states_[opponent_states_.size() - 2]] = 1; 
    // std::cout << "ballstate: " << ball_states_[ball_states_.size() - 2] << std::endl;
    // std::cout << "opponentstates: " << opponent_states_[opponent_states_.size() - 2] << std::endl;
    std::vector<tiny_dnn::vec_t> input;
    input.push_back(stateVector);
    // Create output
    tiny_dnn::vec_t output;
    output = qValues_[qValues_.size() - 2];
    // std::cout << "new q: " << reward + GAMMA * *std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end()) << std::endl;
    output[actions_[actions_.size() - 2]] += reward + GAMMA *(*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end()) - output[actions_[actions_.size() - 2]]);
    /*Q-Learning的更新公式，*/
    // std::cout << "action: " << actions_[actions_.size() - 2] << std::endl;
    std::vector<tiny_dnn::vec_t> qValues;
    qValues.push_back(output);
    // std::cout << "---------- nn TRAINING ----------" << std::endl;
    tiny_dnn::gradient_descent opt;//梯度下降
    nn.fit<tiny_dnn::mse>(opt, input, qValues, BATCH_SIZE, EPOCHS);
    // std::cout << "---------- nn TRAINING done----------" << std::endl;
  }
}
//写得分episode，xPositions_,yPositions_，qvalue，lastaction，和nn
void BehaviorLearnersa::writeToFile(){
  std::cout << "start to write\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "neural_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::ofstream outputStream(modelFilePath);
  outputStream << nn;
  // write evaluation information to file
  std::cout << "write nn finish\n";
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
//把action以添加的方式写入文件
void BehaviorLearnersa::writeActionToFile(){
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

void BehaviorLearnersa::write100rate(float rate){
  std::string rate100filename = resultsFolder + "high_rate100.csv";
  std::ofstream outFile;  
  outFile.open(rate100filename, std::ios::app);
  outFile << rate << ",";
  outFile.close();
}
void BehaviorLearnersa::writescoreepisode(int scoreepisode){
  std::string scoreepisodefilename = resultsFolder + "high_scoreepisode.csv";
  std::ofstream outFile;  
  outFile.open(scoreepisodefilename, std::ios::app);
  outFile << scoreepisode << ",";
  outFile.close();
}
void BehaviorLearnersa::writescorestair(int scorestair){
  std::string scorestairfilename = resultsFolder + "high_scorestair.csv";
  std::ofstream outFile;  
  outFile.open(scorestairfilename, std::ios::app);
  outFile << scorestair << ",";
  outFile.close();
}
void BehaviorLearnersa::writeepisodereward(int episodereward){
  std::string episoderewardfilename = resultsFolder + "high_episodereward.csv";
  std::ofstream outFile;  
  outFile.open(episoderewardfilename, std::ios::app);
  outFile << episodereward << ",";
  outFile.close();
}
void BehaviorLearnersa::writecollectedreward(int collectedreward){
  std::string collectedrewardfilename = resultsFolder + "high_collectedreward.csv";
  std::ofstream outFile;  
  outFile.open(collectedrewardfilename, std::ios::app);
  outFile << collectedreward << ",";
  outFile.close();
}

void BehaviorLearnersa::initQTable(){
  if (TRAINING && !LOAD_FROM_FILE) {
  } else {
    std::string fileName = resultsFolder + "q_9_9";
    std::cout << "Loaded from " << fileName << std::endl;
    File modelFile(fileName, "r");
    std::string modelFilePath = modelFile.getFullName();
    std::cout << "Loaded from " << modelFilePath << std::endl;
    std::ifstream inputStream(modelFilePath);
    for (int i = 0; i < 9*9; ++i) {
      for (int j = 0; j < 2; ++j) {
        inputStream >> QTable[i][j];
      }
    }
  }
  std::cout << std::endl;
}
void BehaviorLearnersa::updateQTable(float reward){
  epireward+=reward;
  std::cout << "updating QTable using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
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
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*9+opponentstate][action] = output[actions_[i]];
      updateWeight *= 0.7;
    }
  }
}
void BehaviorLearnersa::updateQTableAfterGoal(float reward){
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
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*9+opponentstate][action] = output[actions_[i]];
      updateWeight *= 0.7;
    }
  }
}
void BehaviorLearnersa::updateQTableforFirstAction(float reward){
  epireward+=reward;
  // std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
  if (actions_.size() == 1) {
    int ballstate = ball_states_[ball_states_.size() - 1];
    int opponentstate = opponent_states_[opponent_states_.size() - 1];
    int action = actions_[actions_.size() - 1];
    QTable[ballstate*9+opponentstate][action] += reward;
  }
}
void BehaviorLearnersa::updateQTablelamda(float reward){
  epireward+=reward;
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
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersa::updateQTableAfterGoallamda(float reward){
  epireward+=reward;
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
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersa::updateQTableforFirstActionlamda(float reward){
  epireward+=reward;
  // std::cout << "updating QTable for 1st action using reward ---> " << reward << std::endl;
  if (actions_.size() == 1) {
    std::cout << "updating QTable lambda for 1st action using reward ---> " << reward << std::endl;
    int ballstate = ball_states_[ball_states_.size() - 1];
    int opponentstate = opponent_states_[opponent_states_.size() - 1];
    int action = actions_[actions_.size() - 1];
    QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] += ALPHA * reward;
  }
}

void BehaviorLearnersa::updateQTablesarsa(float reward){
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

    // float delta = reward + GAMMA* (*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) - output[actions_[actions_.size() - 2]];
    for (int i = ball_states_.size() - 2; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
  std::cout << "update finish\n";
}
void BehaviorLearnersa::updateQTableAfterGoalsarsa(float reward){
  epireward+=reward;
  std::cout << "updating sarsa QTable after goal using reward ---> " << reward << std::endl;
  if (ball_states_.size() > 1) {
    //n-step back
    std::vector<tiny_dnn::vec_t> input;
    tiny_dnn::vec_t output;
    float updateWeight = LAMDA;
    output = qValues_[ ball_states_.size() - 1 ];
    float delta = reward;
    // float delta = reward + GAMMA* (*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) - output[actions_[actions_.size() - 2]];
    for (int i = ball_states_.size() - 1; i >= 0; --i) {
      // Create input
      tiny_dnn::vec_t stateVector(AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES, 0);
      stateVector[ball_states_[i]] = 1;
      stateVector[AMOUNT_BALL_STATES + opponent_states_[i]] = 1;
      input.push_back(stateVector);
      // Create output
      output = qValues_[i];
      output[actions_[i]] += ALPHA * delta * updateWeight;
      int ballstate = ball_states_[i];
      int opponentstate = opponent_states_[i];
      int action = actions_[i];
      QTable[ballstate*AMOUNT_OPPONENT_STATES+opponentstate][action] = output[actions_[i]];
      updateWeight *= GAMMA * LAMDA;
    }
  }
}
void BehaviorLearnersa::updateQTableforFirstActionsarsa(float reward){
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



void BehaviorLearnersa::selectActionFromeQTable(){
  int stateVector[AMOUNT_BALL_STATES + AMOUNT_OPPONENT_STATES] = {0};
  stateVector[ball_states_.back()] = 1;
  stateVector[AMOUNT_BALL_STATES + opponent_states_.back()] = 1;
  // Predict qValues
  int ballstate = ball_states_.back();
  int opponentstate = opponent_states_.back();

  // tiny_dnn::vec_t qValues = nn.predict(stateVector);
  tiny_dnn::vec_t qValues;
  qValues.push_back(QTable[ballstate*9+opponentstate][0]);
  qValues.push_back(QTable[ballstate*9+opponentstate][1]);
  int bestAction = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
  if (!TRAINING) {
    // int bestAction = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
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
    if(fabs(qValues[i]) < 0.00001)
      qValues[i] = 0;
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
    actions_.push_back(bestAction);
  }
  // actions_.push_back(d(gen));
  qValues_.push_back(qValues);
  std::cout << "BehaviorLearner Select Actions" << std::endl;
  std::cout << " you should take (" << actions_.back() << ")th action" << std::endl;

}
void BehaviorLearnersa::writeQTableToFile(){
  // std::cout << "start to write\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "q_" + std::to_string(AMOUNT_BALL_STATES) + "_" + std::to_string(AMOUNT_OPPONENT_STATES);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::cout << "start to write-->"<<modelFilePath<<std::endl;
  std::ofstream outputStream(modelFilePath);
  for (int i = 0; i < 9*9; ++i) {
    for (int j = 0; j < 2; ++j) {
      outputStream << QTable[i][j] << " ";
    }
  }
}


void BehaviorLearnersa::writeScoreAndPosition(){
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


MAKE_MODULE(BehaviorLearnersa, behaviorControl)
