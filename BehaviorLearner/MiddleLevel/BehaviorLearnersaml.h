/*
 * BehaviorLearnersaml.h
 * @haochen2115
 */

#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wunused-private-field"
#include "tiny_dnn/tiny_dnn.h"
#pragma clang diagnostic pop
#include "Platform/File.h"
#include "Platform/Time.h"
#include "Representations/Configuration/FieldDimensions.h"
#include "Representations/Infrastructure/GameInfo.h"
#include "Representations/Modeling/BallModel.h"
#include "Representations/Modeling/ObstacleModel.h"
#include "Representations/Modeling/RobotPose.h"
#include "Representations/BehaviorControl/BehaviorAction.h"
#include "Representations/Infrastructure/TeamInfo.h"
#include "Tools/Module/Module.h"
#include "Tools/Math/Pose2f.h"
#include "Tools/Debugging/ColorRGBA.h"
#include "Tools/Debugging/DebugDrawings.h"
#include <fstream>
#define CLOSE_DISTANCE 500

// Update way
// #define N_STEP_BACK_TD
// #define Q_LAMDA
#define SARSA

#define QTABLE
// #define DQN

MODULE(BehaviorLearnersaml,
{,
  REQUIRES(BallModel),
  REQUIRES(FieldDimensions),
  REQUIRES(GameInfo),
  REQUIRES(ObstacleModel),
  REQUIRES(OpponentTeamInfo),
  REQUIRES(OwnTeamInfo),
  REQUIRES(RobotPose),
  PROVIDES(BehaviorAction),
});

class BehaviorLearnersaml: public BehaviorLearnersamlBase {
public:
  BehaviorLearnersaml();

  void update(BehaviorAction& behaviorAction);

private:

  bool start = false; //比赛是否开始，时间处于0-600才开始训练并update
  static const bool TRAINING = true;//是否训练，如果训练完毕需要测试，将这个置为false
  static const bool LOAD_FROM_FILE = true;//不训练，或者接着上一次的结果训练，置为false
  static const int AMOUNT_BALL_STATES = 54;
  static const int AMOUNT_OPPONENT_STATES = 54;
  static const int AMOUNT_STATES = AMOUNT_BALL_STATES * AMOUNT_OPPONENT_STATES;
  static const int AMOUNT_ACTIONS = 17;
  static const int OPPONENT_FIRST_LEVEL_DISTANCE = 500;//划分状态的第一圈，第一圈内过人
  static const int OPPONENT_SECOND_LEVEL_DISTANCE = 1000;//划分状态的第二圈，第二圈及其外shoot或者dribble
  constexpr static const float TAU = 0.7f;//softmax分布选择动作时的温度
  constexpr static const float GAMMA = 0.99f;//Q-learning的参数，更新时给下个状态/动作的权重
  constexpr static const float LAMDA = 0.9f;
  constexpr static const float ALPHA = 0.4f;//update step
  size_t BATCH_SIZE = 1;//神经网络更新的堆大小
  size_t EPOCHS = 500;//每次神经网络更新的回合限制
  tiny_dnn::network<tiny_dnn::sequential> nn;//根据状态预测qvalue以预测动作的神经网络

  std::vector<int> ball_states_;//每回合的球的所有状态
  std::vector<int> opponent_states_;//每回合的对方球员的所有状态
  std::vector<int> actions_;//每回合的所有动作
  std::vector<int> hasreward_;//是否已经被惩罚或者奖励过
  std::vector<tiny_dnn::vec_t> qValues_;//每回合计算出来的所有q值
  std::vector<int> xPositions_;//记录机器人轨迹的x坐标
  std::vector<int> yPositions_;//记录机器人轨迹的x坐标
  int ownScore = 0;//己方得分
  int opponentScore = 0;//敌方得分
  int amountTrials = 1;//回合次数
  int startTime;//开始时间
  int goalsScoredInRow = 0;//连续得分次数
  bool lastAttemptGoal = false;//上一次是否进球，为了记录连续得分
  std::vector<int> scoredInEpisodes;//记录得分的回合
  std::string resultsFolder;//存储结果的文件夹路径
  std::string resultsFolderAddr;//用于输出在终端的文件夹路径字符串
  //表征状态的一些变量
  int ballstatebin;
  int opponentstatebin;
  bool goalbyblue,goalbyred;
  int intermediatereward;
  int terminalreward;
  bool balliscontrolledbyopponent;
  bool ballisbehindme;

  int discretizeBallPosition(Vector2f ballposition);//离散化球的区域，返回1-9
  int discretizeOpponentPosition(Vector2f opponentposition);//离散化对方球员的区域，返回1-9
  void getState();//获取球和对方球员的状态,更新ballstatebin,opponentstatebin;
  void getGameScene();//获取用于给奖赏的比赛状态,更新balliscontrolledbyopponent;更新intermediatereward
  void getGameState();//获取比赛是否结束的状态,更新endofepisode和terminalreward;更新goalbyblue,goalbyred,goalsScoredInRow;
  void writeToFile();//写网络和位置信息，得分信息
  void writeActionToFile();//写动作序列

  float QTable[AMOUNT_BALL_STATES*AMOUNT_OPPONENT_STATES][AMOUNT_ACTIONS] = {0};
  void initQTable();
  void updateQTable(float reward);
  void updateQTableforFirstAction(float reward);
  void updateQTableAfterGoal(float reward);
  void updateQTablelamda(float reward);
  void updateQTableforFirstActionlamda(float reward);
  void updateQTableAfterGoallamda(float reward);
  void updateQTablesarsa(float reward);
  void updateQTableforFirstActionsarsa(float reward);
  void updateQTableAfterGoalsarsa(float reward);
  void selectActionFromeQTable();
  void writeQTableToFile();
  void writeScoreAndPosition();
  void updateQNetwork();//状态数小于miniBatch个就不更新；否则从记忆池里选择miniBatch个更新(记忆池size小于500,则按最大容量选)，每次更新，但每C次替换一次原来的QNN；
  void putStateintoMemoryPool(float reward);//没达到CAPACITY之前，一直push_back，如果memory的size达到容量了，就从当前position开始，一一往后覆盖；
  void putStateintoMemoryPoolfirsttime(float reward);
  static const int MEMORY_POOL_CAPACITY = 500;
  static const int miniBatch = 50;
  static const int C_STEP_CHANGE = 50;
  static const int QUANTITY_STATES = 2;
  static const int QUANTITY_ACTIONS = 17;
  int position = 0;//position加到CAPACITY后清零
  int update_count = 0;
  std::vector<int> memory_ball_states_;//每次压入ball_states_的时候把上一次的状态压入memory;
  std::vector<int> memory_opponent_states_;
  std::vector<int> memory_actions_;
  std::vector<int> memory_reward_;
  
  std::vector<int> memory_num;
  std::vector<int> sample_num;

  std::vector<tiny_dnn::vec_t> memory_input;
  std::vector<tiny_dnn::vec_t> memory_qValues;
  tiny_dnn::network<tiny_dnn::sequential> nn_update;
  void initQNetwork();//初始化成State-Action-Action-qvalue;RBF(State-RBF-Action)
  void writeQNetworkToFile();
  void sampleRandomTrainBatch();
  void writeLatestQNetworkToFile();
  void selectActionFromQNetwork();
  int epireward = 0;
  void write100rate(float rate);
  void writescoreepisode(int scoreepisode);
  void writescorestair(int scorestair);
  void writeepisodereward(int episodereward);
  void writecollectedreward(int collectedreward);
  
};
