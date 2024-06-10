/*
 * DribbleLearner.cpp
 * @haochen2115
 */

#include "DribbleLearner.h"

#include <iostream>
DribbleLearner::DribbleLearner() {
  if (theOwnTeamInfo.teamColor == TEAM_BLUE) {
    startTime = Time::getRealSystemTime();
    File pathFile("resultPath", "r");
    std::string filePath = pathFile.getFullName();
    std::ifstream inputStream(filePath);
    inputStream >> resultsFolder;
    // initNetwork();
    initSpeedQ();//初始化或者加载Q表
  // getState();
  // getGameScene();
  // getGameState();
    update_count = 0;
    resultsFolderAddr = "Time@" + std::to_string(startTime) + " | "  + "resultsFolderAddr is -> " + resultsFolder;
    std::cout << resultsFolderAddr << std::endl;
    std::cout << "initialized" << std::endl;
    DECLARE_PLOT("module:DribbleLearner:gamereward");
    writeparameter();
  }
}
void DribbleLearner::update(BehaviorAction& behaviorAction) {
  DECLARE_PLOT("module:DribbleLearner:gamereward");
  PLOT("module:DribbleLearner:gamereward", gamereward); 

  if (theOwnTeamInfo.teamColor == TEAM_RED) return;
  if (theGameInfo.secsRemaining > 0 && theGameInfo.secsRemaining < 600) {
    start = true;
  }
  if (!start) return;
  //最开始的对球等待
  if(alignmentdelay < 400){
  	alignmentdelay ++;
  	behaviorAction.behavior_action = 6;
  	return;
  }else if(alignmentdelay == 400){
    std::cout << "alignment maybe finish now..." << std::endl;
    alignmentdelay++;
	}
  #ifdef ROUND
  if(pleaseturnaround > 0 && overleft){
    pleaseturnaround --;
    if(pleaseturnaround > 1000 )
      behaviorAction.behavior_action = 9;
    else
      behaviorAction.behavior_action = 7;//turn for 2000,0
    // std::cout << "overleft and turn right\n";
    return;
  }
  else if(pleaseturnaround > 0 && overright){
    pleaseturnaround --;
    if(pleaseturnaround > 1000 )
      behaviorAction.behavior_action = 9;
    else
      behaviorAction.behavior_action = 8;//turn for -2000,0
    // std::cout << "overright and turn left\n";
    return;
  }
  #endif
  //然后计数定时更新
  #define UPDATECOUNT 30
  update_count++;
  update_count = update_count == UPDATECOUNT ? 0: update_count;
  if(update_count != UPDATECOUNT-1) return;//上次感觉50接近于1秒

  //对球有问题，中间阶段的对球
  Vector2f ballpositionRobot = theBallModel.estimate.position;
  if( !(ballpositionRobot.x() < 600.f) )
  {
    punishonce = true;
    // std::cout << "ballpositionRobot.x() ---> " << ballpositionRobot.x() << std::endl;
    #ifdef GOAL
    behaviorAction.behavior_action = 6;//please alignment;
    #endif
    #ifdef ROUND
    if(checkdirection == true)
      behaviorAction.behavior_action = 8;//please alignment;
    else
      behaviorAction.behavior_action = 7;
    #endif
    // faultstime++;
    return;
  }
  if(! (fabs(ballpositionRobot.y()) < 100.f) )
  {
    // faultstime++;
    // punishonce = true;
    // std::cout << "ballpositionRobot.y() ---> " << ballpositionRobot.y() << std::endl;
    #ifdef GOAL
    behaviorAction.behavior_action = 6;//please alignment;
    #endif
    #ifdef ROUND
    if(checkdirection == true)
      behaviorAction.behavior_action = 8;//please alignment;
    else
      behaviorAction.behavior_action = 7;
    #endif
    return;
  }

  if(punishonce == true && TRAINING){
    #ifdef N_STEP_BACK_TD
  	 updateSpeedQ(-10.f);
    #endif
    #ifdef Q_LAMDA
     updateSpeedQlamda(-10.f);
    #endif
    #ifdef SARSA_LAMDA
     updateSpeedSARSAlamda(-10.f);
    #endif
  	punishonce = false;
  }

  Vector2f ball = theBallModel.estimate.position;
  getDistanceState(ball);//获取距离，更新ballrobotdistance；和bin
  getSpeedNow();//根据上一个动作get现在的speed；如果size是0,速度设置为20mm/s;更新robotspeed；和bin
  #ifdef GOAL
    getGameState();//看一下是否结束，更新goalbyblue,goalbyred;
  #endif
  #ifdef ROUND
    getRealGameState(ball);
  #endif

  #ifdef GOAL
  if (goalbyblue || goalbyred)
  #endif
  #ifdef ROUND
  if(overleft || overright)
  #endif
    {
    writeepisodereward();
    episodereward = 0;
    if(TRAINING)
      writeSpeedQTableToFile();//只是写q表

    actionsize.push_back(actions_.size());
    // std::cout << "faultstime" << faultstime << std::endl;
    faultsrate.push_back((float)((float)faultstime/(float)actions_.size()));
    faultsfreq.push_back(faultstime);
    faultsdistancerate = ((float)((float)faultsdistance/(float)actions_.size()));
    // DECLARE_PLOT("module:DribbleLearner:actionsize");
    // DECLARE_PLOT("module:DribbleLearner:faultsrate");
    std::cout << "actionsize-->" << actionsize.back()<<std::endl;
    std::cout << "faultsrate-->" << faultsrate.back()<<std::endl;
    std::cout << "faultsdistancerate--> " << faultsdistancerate << std::endl;
    std::cout << "faultsfreq-->" << faultsfreq.back()<<std::endl;
    std::cout << "faultsdistance-->" << faultsdistance<<std::endl;
    
    writeactionsize(actionsize.back());
    writefaultsrate(faultsrate.back());
    writefaultsfreq(faultsfreq.back());
    writefaultsdistance();
    writefaultsdistancerate();
    faultstime = 0;
    faultsdistance = 0;

    distance_states_.clear();
    actions_.clear();
    qValues_.clear();
    speedaction_.clear();

    ++amountTrials;
    #ifdef GOAL
    start = false;
    alignmentdelay = 0;
    #endif
    #ifdef ROUND
    pleaseturnaround = 1200;
    #endif
  }
  else{
  	distance_states_.push_back(distancebin);
    #ifdef REWARD1
  	 speedreward = getSpeedReward();//否则根据距离是否大于最大距离350，或者速度小于期望速度80给-1；
    #endif
    #ifdef REWARD2
     speedreward = getSpeedReward2();
    #endif
  	selectSpeedAction();//根据当前状态选择动作压入action_
  	// updateSpeedQ(speedreward);//4 step back 吧
    if(actions_.size() > 0){
    //speed1
    #ifdef AT
    changeActionToSpeed();
    #endif
    }
    if(TRAINING){
    #ifdef N_STEP_BACK_TD
     updateSpeedQ(speedreward);
    #endif
    #ifdef Q_LAMDA
     updateSpeedQlamda(speedreward);
    #endif
    #ifdef SARSA_LAMDA
     updateSpeedSARSAlamda(speedreward);
    #endif
    }
  }
  if (TRAINING && amountTrials == 3000) {
    std::cout << "Done training, stopping program." << std::endl;
    exit(EXIT_SUCCESS);
  }
  if(actions_.size() > 0){
    //speed1
    #ifdef AT
    // changeActionToSpeed();
    behaviorAction.behavior_action = speedaction_.back();
    #endif
    #ifdef VT
    behaviorAction.behavior_action = actions_.back();
    #endif
  }
}
void DribbleLearner::changeActionToSpeed()
{
  if(speedaction_.size() == 0)
    speedaction_.push_back(0);
  else
  {
    if(actions_.back() == 0)//减速
    {
      if(speedaction_.back() == 0)
        speedaction_.push_back(0);
      else
        speedaction_.push_back(speedaction_.back() - 1);
    }
    else if(actions_.back() == 1)
    {
      speedaction_.push_back(speedaction_.back());
    }
    else
    {
      if(speedaction_.back() == 4)
        speedaction_.push_back(4);
      else
        speedaction_.push_back(speedaction_.back() + 1);
    }
  }
}
void DribbleLearner::initSpeedQ(){
  if (TRAINING && !LOAD_FROM_FILE) {
  } else {
    // Read network from file
    std::string fileName = resultsFolder + "speed_q_" + std::to_string(AMOUNT_DISTANCE_STATES) + "_" + std::to_string(AMOUNT_ACTIONS);
    std::cout << "Loaded from " << fileName << std::endl;
    File modelFile(fileName, "r");
    std::string modelFilePath = modelFile.getFullName();
    std::cout << "Loaded from " << modelFilePath << std::endl;
    std::ifstream inputStream(modelFilePath);
    for (int i = 0; i < AMOUNT_DISTANCE_STATES; ++i) {
      for (int j = 0; j < AMOUNT_ACTIONS; ++j) {
        inputStream >> QTable[i][j];
      }
    }
  }
  for (int i = 0; i < AMOUNT_DISTANCE_STATES; ++i) {
    for (int j = 0; j < AMOUNT_ACTIONS; ++j) {
      std::cout << "qv: " << QTable[i][j] << std::endl;
    }
  }
  std::cout << std::endl;
}
void DribbleLearner::getDistanceState(Vector2f ball){
	if(ball.norm() >= 500){
    distancenow = 10;
		distancebin = 10;
  }
	else{
    distancenow = ball.norm();
		distancebin = (int)(ball.norm()/50);
  }
}
void DribbleLearner::getSpeedNow(){
	if(actions_.size() > 0){
    #ifdef VT
		speedbin = actions_.back();
    speednow = (actions_.back() + 1) * 20;
    #endif
    //speed2
    #ifdef AT
    speedbin = speedaction_.back();
    speednow = (speedaction_.back() + 1) * 20;
    #endif
  }
	else{
		speedbin = 0;
    speednow = 20;
  }
}
void DribbleLearner::getGameState(){
  if (theOwnTeamInfo.score > ownScore) {//如果得分的话，奖赏为剩余时间奖励加2,嗯所以一旦进球，奖励就会大于2
    ++ownScore;
    goalbyblue = true;
  } else if (theOpponentTeamInfo.score > opponentScore) {
    ++opponentScore;
    goalbyred = true;
  }
  else
  {
    goalbyblue = false;
    goalbyred = false;
  }
  // Set score back to 0 when 255 because unsigned char
  if (ownScore == 255 && theOwnTeamInfo.score == 0) {
    ownScore = 0;
    goalbyblue = true;
  }
  else if (opponentScore == 255 && theOpponentTeamInfo.score == 0) {
    opponentScore = 0;
    goalbyred = true;
  }
}
void DribbleLearner::getRealGameState(Vector2f ball){
  overright = false;
  overleft = false;
  Vector2f ballfield = theRobotPose*ball;
  if(ballfield.x() > 2000.f && checkdirection == false){
    overright = true;
    checkdirection = true;
  }
  else if(ballfield.x() < -2000.f && checkdirection == true){
    overleft = true;
    checkdirection = false;
  }
}
void DribbleLearner::writeSpeedQTableToFile(){
  std::cout << "start to write\n";
  // write network to file
  std::string neuralFileName = resultsFolder + "speed_q_" + std::to_string(AMOUNT_DISTANCE_STATES) + "_" + std::to_string(AMOUNT_ACTIONS);
  File modelFile(neuralFileName, "w");
  std::string modelFilePath = modelFile.getFullName();
  std::ofstream outputStream(modelFilePath);
  for (int i = 0; i < AMOUNT_DISTANCE_STATES; ++i) {
    for (int j = 0; j < AMOUNT_ACTIONS; ++j) {
      outputStream << QTable[i][j] << " ";
    }
  }
  // write evaluation information to file
  std::cout << "write qtable finish\n";
  int elapsedTime = Time::getRealTimeSince(startTime);
  std::cout << "#" << amountTrials << " amount of trials." << std::endl;
  std::cout << "This took " << elapsedTime/1000 << "seconds (" << (elapsedTime/1000)/60 << "minutes)" << std::endl;
}
void DribbleLearner::writeSpeedState(){
  std::string actionFileName = resultsFolder + "distance_" + std::to_string(AMOUNT_DISTANCE_STATES) + "_" + std::to_string(AMOUNT_ACTIONS) + "_" + std::to_string(amountTrials);
  File modelFile(actionFileName, "w");
  std::string actionFilePath = modelFile.getFullName();
  std::ofstream os(actionFilePath);
  os << "distance_states --> refer to .csv file" << "\n";
  std::ofstream outFile;  
  outFile.open(actionFilePath + ".csv", std::ios::app);

  std::cout << "distance_states ---> " << distance_states_.size() << std::endl;
  if(distance_states_.size() > 0){
    for (int distance_states : distance_states_) {
      outFile << distance_states << ", ";
    }
    outFile << "\n";
  }
  outFile.close(); 
}
void DribbleLearner::writeSpeedAction(){
  std::string actionFileName = resultsFolder + "actions_" + std::to_string(AMOUNT_DISTANCE_STATES) + "_" + std::to_string(AMOUNT_ACTIONS) + "_" + std::to_string(amountTrials);
  File modelFile(actionFileName, "w");
  std::string actionFilePath = modelFile.getFullName();
  std::ofstream os(actionFilePath);
  os << "actions_ --> refer to .csv file" << "\n";
  std::ofstream outFile;  
  outFile.open(actionFilePath + ".csv", std::ios::app);

  std::cout << "actions_ ---> " << actions_.size() << std::endl;
  if(actions_.size() > 0){
    for (int actions : actions_) {
      outFile << actions << ", ";
    }
    outFile << "\n";
  }
  outFile.close(); 
}
void DribbleLearner::writeactionsize(int wactionsize){
  std::string actionsizefilename = resultsFolder + "dribble_actionsize.csv";
  std::ofstream outFile;  
  outFile.open(actionsizefilename, std::ios::app);
  // std::cout << "actionsize--->"<< wactionsize << std::endl;
  outFile << wactionsize << ",";
  outFile.close();
}
void DribbleLearner::writefaultsrate(float wfaultsrate){
  std::string faultsratefilename = resultsFolder + "dribble_faultsrate.csv";
  std::ofstream outFile;  
  outFile.open(faultsratefilename, std::ios::app);
  // std::cout << "faultsrate--->"<< wfaultsrate << std::endl;
  outFile << wfaultsrate << ",";
  outFile.close();
}
void DribbleLearner::writefaultsfreq(int faultsfreq){
  std::string faultsfreqfilename = resultsFolder + "dribble_faultsfreq.csv";
  std::ofstream outFile;  
  outFile.open(faultsfreqfilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << faultsfreq << ",";
  outFile.close();
}
void DribbleLearner::writefaultsdistance(){
  std::string faultsdistancefilename = resultsFolder + "dribble_faultsdistance.csv";
  std::ofstream outFile;  
  outFile.open(faultsdistancefilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << faultsdistance << ",";
  outFile.close();
}
void DribbleLearner::writefaultsdistancerate(){
  std::string faultsdistanceratefilename = resultsFolder + "dribble_faultsdistancerate.csv";
  std::ofstream outFile;  
  outFile.open(faultsdistanceratefilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << faultsdistancerate << ",";
  outFile.close();
}
void DribbleLearner::writegamereward()
{
  std::string gamerewardfilename = resultsFolder + "dribble_gamereward.csv";
  std::ofstream outFile;  
  outFile.open(gamerewardfilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << gamereward << ",";
  outFile.close();
}
void DribbleLearner::writeepisodereward()
{
  std::string episoderewardfilename = resultsFolder + "dribble_episodereward.csv";
  std::ofstream outFile;  
  outFile.open(episoderewardfilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << episodereward << ",";
  outFile.close();
}
void DribbleLearner::writeparameter()
{
  std::string parameterfilename = resultsFolder + "parameter";
  std::ofstream outFile; 
  outFile.open(parameterfilename, std::ios::app);
  // std::cout << "faultsfreq--->"<< faultsfreq << std::endl;
  outFile << "GAMMA" << " : " << GAMMA << "\n";
  outFile << "ALPHA" << " : " << ALPHA << "\n";
  outFile << "LAMDA" << " : " << LAMDA << "\n";
  outFile << "TAU" << " : " << TAU << "\n";
  // outFile << "gamma" << " : " << GAMMA << "\n";
  // outFile << "gamma" << " : " << GAMMA << "\n";
  #ifdef N_STEP_BACK_TD
  outFile << "using n-step back td \n";
  #endif
  #ifdef Q_LAMDA
  outFile << "using q lambda \n";
  #endif
  #ifdef SARSA_LAMDA
  outFile << "using sarsa lambda \n";
  #endif
  #ifdef REWARD1
  outFile << "using REWARD1 \n";
  outFile << " 8/-10 2/-8 10 \n";
  // outFile << "8/-10  -1/-8 10 \n";
  #endif
  #ifdef REWARD2
  outFile << "using REWARD2 \n";
  #endif
  #ifdef AT
  outFile << "using AT \n";
  #endif
  outFile << "update_count : " << UPDATECOUNT << "\n";

  #ifdef GOAL
  outFile << "GOAL MODE\n";
  #endif
  #ifdef ROUND
  outFile << "ROUND MODE\n";
  #endif
  outFile.close();
}

float DribbleLearner::getSpeedReward(){
	if(distancebin >= 8)//7/8/9/10
	{
		std::cout << "too far\n";
    faultstime++;
    faultsdistance++;
		return -10;
	}
	else if(speedbin < 2)//(3#-10)
	{
		std::cout << "too slow\n";
    faultstime++;
		return -8;
	} 
	else
	{
		std::cout << "good\n";
		return 10;
	}
}
float DribbleLearner::getSpeedReward2(){
  float continuous_reward;
  continuous_reward = -(float)distancenow/(float)speednow;
  if(distancenow <= 350 && speednow > 60)
    return 5.f;
  return continuous_reward;
}
void DribbleLearner::selectSpeedAction(){
	tiny_dnn::vec_t qValues;
	for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
    #ifdef AT
    statebin = distancebin*SPEED_STATE+speedbin;
    qValues.push_back(QTable[statebin][i]);
    #endif
    #ifdef VT
    qValues.push_back(QTable[distancebin][i]);
    #endif
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
	  probabilities[i] = exp((double)(qValues[i] - maxQValue) / fabs(maxQValue) * TAU);//!!!
	else
	  probabilities[i] = exp((qValues[i]) / TAU);
	sumProbabilities += probabilities[i];
	std::cout << "qvalue: " << qValues[i] << std::endl;
	// std::cout << "e: " << probabilities[i] << std::endl;    
	}
	// std::cout << "sumProbabilities: " << sumProbabilities << std::endl;
	for (int i = 0; i < AMOUNT_ACTIONS; ++i) {
	probabilities[i] /= sumProbabilities;
	// std::cout << "prob " << i << ": " << probabilities[i] << std::endl;
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
void DribbleLearner::updateSpeedQ(float reward){
	std::cout << "updating QTable using reward ---> " << reward << std::endl;
	if(distance_states_.size() > 1){
		std::vector<tiny_dnn::vec_t> input;
	    tiny_dnn::vec_t output;
	    tiny_dnn::vec_t qValues;
	    float updateWeight = 1;
	    for (int i = distance_states_.size() - 2; i >= distance_states_.size() - 6; --i) {//4s tep back
	      if(i < 0) break;
	      // Create output
	      output = qValues_[i];
	      output[actions_[i]] += (reward * updateWeight);
        #ifdef VT
	      int distancestatebin = distance_states_[i];
	      int action = actions_[i];
	      QTable[distancestatebin][action] = output[actions_[i]];
        #endif
        #ifdef AT
        int distancestatebin = distance_states_[i];
        int speedstatebin = speedaction_[i];
        int distancespeedbin = distancestatebin*SPEED_STATE+speedstatebin;
        int action = actions_[i];
        QTable[distancespeedbin][action] = output[actions_[i]];
        #endif
	      updateWeight *= 0.7;
	    }
	}
}
void DribbleLearner::updateSpeedQlamda(float reward){
  std::cout << "updateing Q lamda using reward ---> " << reward << std::endl;
  if(distance_states_.size() > 1 && actions_.size() > 1 && qValues_.size() > 0){
    std::vector<tiny_dnn::vec_t> input;
      tiny_dnn::vec_t output;
      tiny_dnn::vec_t qValues;
      float updateWeight = LAMDA;
      output = qValues_[distance_states_.size() - 2];
      float delta = reward + GAMMA* (*std::max_element(qValues_[qValues_.size() - 1].begin(), qValues_[qValues_.size() - 1].end())) - output[actions_[actions_.size() - 2]];
      // std::cout << "delta is done\n";s
      for (int i = distance_states_.size() - 2; i >= distance_states_.size() - 6; --i) {//4s tep back
        if(i < 0) break;
        // Create output
        output = qValues_[i];
        output[actions_[i]] += ALPHA * delta * updateWeight;

        // int distancestatebin = distance_states_[i];
        // int action = actions_[i];
        // QTable[distancestatebin][action] = output[actions_[i]];、

        if(distance_states_.size() != actions_.size())
          std::cout << "ERROR-------------------------------!" << std::endl;

        #ifdef VT
        int distancestatebin = distance_states_[i];
        int action = actions_[i];
        QTable[distancestatebin][action] = output[actions_[i]];
        #endif
        #ifdef AT
        int distancestatebin = distance_states_[i];
        int speedstatebin = speedaction_[i];
        int distancespeedbin = distancestatebin*SPEED_STATE+speedstatebin;
        int action = actions_[i];
        QTable[distancespeedbin][action] = output[actions_[i]];
        #endif

        updateWeight *= GAMMA * LAMDA;
      }
  }
  else
    std::cout << "unexpected size occurred\n";
}
void DribbleLearner::updateSpeedSARSAlamda(float reward){
  gamereward += reward;
  writegamereward();
  episodereward += reward;
  std::cout << "updateing SARSA lamda using reward ---> " << reward << std::endl;
  if(distance_states_.size() > 1){
    std::vector<tiny_dnn::vec_t> input;
      tiny_dnn::vec_t output;
      tiny_dnn::vec_t qValues;
      qValues = qValues_[qValues_.size() - 2];
      output = qValues_[distance_states_.size() - 2];
      float updateWeight = LAMDA;
      float delta = reward + GAMMA* (qValues[actions_[actions_.size() - 1]]) - output[actions_[actions_.size() - 2]];
      for (int i = distance_states_.size() - 2; i >= distance_states_.size() - 6; --i) {//4s tep back
        if(i < 0) break;
        // Create output
        output = qValues_[i];
        output[actions_[i]] += ALPHA * delta * updateWeight;

        // int distancestatebin = distance_states_[i];
        // int action = actions_[i];
        // QTable[distancestatebin][action] = output[actions_[i]];
        if(distance_states_.size() != actions_.size())
          std::cout << "ERROR-------------------------------!" << std::endl;
        #ifdef VT
        int distancestatebin = distance_states_[i];
        int action = actions_[i];
        QTable[distancestatebin][action] = output[actions_[i]];
        #endif
        #ifdef AT
        int distancestatebin = distance_states_[i];
        int speedstatebin = speedaction_[i];
        int distancespeedbin = distancestatebin*SPEED_STATE+speedstatebin;
        int action = actions_[i];
        QTable[distancespeedbin][action] = output[actions_[i]];
        #endif

        updateWeight *= GAMMA * LAMDA;
      }
  }
}

MAKE_MODULE(DribbleLearner, behaviorControl)
