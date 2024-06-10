/*
 * DribbleLearner.h
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
// #include "Representations/BehaviorControl/AlignmentState.h"
#include "Tools/Module/Module.h"
#include "Tools/Math/Pose2f.h"
#include "Tools/Debugging/ColorRGBA.h"
#include "Tools/Debugging/DebugDrawings.h"
#include <fstream>
#define CLOSE_DISTANCE 500

//Update Method
// #define N_STEP_BACK_TD
// #define Q_LAMDA
#define SARSA_LAMDA

//Reward function
#define REWARD1
// #define REWARD2

#define GOAL
// #define ROUND
//game controller also need change

#define VT
// #define AT

MODULE(DribbleLearner,
{,
  REQUIRES(BallModel),
  REQUIRES(FieldDimensions),
  REQUIRES(GameInfo),
  REQUIRES(ObstacleModel),
  REQUIRES(OpponentTeamInfo),
  REQUIRES(OwnTeamInfo),
  REQUIRES(RobotPose),
  //REQUIRES(RoleBehavior),
  // REQUIRES(AlignmentState),
  PROVIDES(BehaviorAction),
});

class DribbleLearner: public DribbleLearnerBase {
public:
  DribbleLearner();

  void update(BehaviorAction& behaviorAction);

private:

  bool start = false; //比赛是否开始，时间处于0-600才开始训练并update
  static const bool TRAINING = true;//是否训练，如果训练完毕需要测试，将这个置为false
  static const bool LOAD_FROM_FILE = true;//不训练，或者接着上一次的结果训练，置为false
  constexpr static const float TAU = 0.7f;//softmax分布选择动作时的温度
  constexpr static const float GAMMA = 0.99f;//Q-learning的参数，更新时给下个状态/动作的权重
  constexpr static const float LAMDA = 0.9f;//eligibility trace~decay rate
  constexpr static const float ALPHA = 0.1f;//update step,Q = Q + Alpha*delta*Eligibility
  int ownScore = 0;//己方得分
  int opponentScore = 0;//敌方得分
  int startTime;//开始时间

  std::vector<int> scoredInEpisodes;//记录得分的回合
  std::string resultsFolder;//存储结果的文件夹路径
  std::string resultsFolderAddr;//用于输出在终端的文件夹路径字符串

  #ifdef VT
  static const int AMOUNT_DISTANCE_STATES = 11;//0,50,100,150,200,250,300,350,400,450,500
  #endif
  #ifdef AT
  static const int AMOUNT_DISTANCE_STATES = 55;
  #endif
  #ifdef VT
  static const int AMOUNT_ACTIONS = 5;//20mm/s,40mm/s,60mm/s,80mm/s,100mm/s
  #endif
  #ifdef AT
  static const int AMOUNT_ACTIONS = 3;
  static const int SPEED_STATE = 5;
  #endif


  int update_count;
  int distancenow;
  int speednow;
  int distancebin;//0~10
  int speedbin;//0~4
  int statebin; // distancebin*5+speedbin
  std::vector<int> distance_states_;
  std::vector<int> actions_;
  std::vector<int> speedaction_;
  std::vector<tiny_dnn::vec_t> qValues_;//每回合计算出来的所有q值
  void getDistanceState(Vector2f ballPosition);
  void getSpeedNow();
  float speedreward;
  float getSpeedReward();
  float getSpeedReward2();
  int amountTrials = 1;
  void getGameState();
  void selectSpeedAction();
  void initSpeedQ();
  void updateSpeedQ(float reward);//4 step back TD
  void updateSpeedQlamda(float reward);//4 step back q(lamda)
  void updateSpeedSARSAlamda(float reward);//4 step back sarsa(lamda)
  void writeSpeedQTableToFile();
  void writeSpeedState();
  void writeSpeedAction();
  float QTable[AMOUNT_DISTANCE_STATES][AMOUNT_ACTIONS] = {0};
  bool goalbyblue,goalbyred;
  int alignmentdelay = 0;
  bool punishonce = false;

  std::vector<int> actionsize;
  std::vector<int> faultsfreq;
  std::vector<float> faultsrate;
  float gamereward=0;
  float episodereward=0;

  int faultstime = 0;
  int faultsdistance = 0;
  void writeactionsize(int wactionsize);
  void writefaultsrate(float wfaultsrate);
  void writefaultsfreq(int faultsfreq);

  void writeepisodereward();
  void writegamereward();

  void writeparameter();
  void writefaultsdistance();

  void getRealGameState(Vector2f ball);
  bool overleft = false;
  bool overright = false;
  bool checkdirection = false;//true-left;false-right
  int pleaseturnaround = 0;
  void writefaultsdistancerate();
  float faultsdistancerate;
  void changeActionToSpeed();

};
