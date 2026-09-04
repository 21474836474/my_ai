#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <mutex>
#include <cctype>
#include <queue>
#include <map>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <windows.h>
#include <thread>
// ========== 新增：CppJieba 头文件 ==========
#include "cppjieba\include\cppjieba\Jieba.hpp"
#include <omp.h>
#include <ctime>
#include <iomanip>
using namespace std;

// ===================== 日志系统 =====================
class Logger {
private:
    ofstream logFile;
    mutex logMtx;
    bool fileEnabled = false;      // 是否写入文件
    bool consoleEnabled = true;    // 是否输出到控制台（默认开启）
    
public:
    Logger() {
        // 先不打开文件，等用户确认后再打开
    }
    
    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    
    // 启用文件日志
    void enable(const string& filename = "log.txt") {
        if (logFile.is_open()) {
            logFile.close();
        }
        logFile.open(filename, ios::out | ios::trunc);
        if (logFile.is_open()) {
            fileEnabled = true;
            cout << "[日志] 文件日志已启用，写入 " << filename << endl;
        } else {
            cout << "[日志] 无法创建日志文件" << endl;
            fileEnabled = false;
        }
    }
    
    // 禁用文件日志
    void disable() {
        fileEnabled = false;
        if (logFile.is_open()) {
            logFile.close();
        }
        cout << "[日志] 文件日志已禁用" << endl;
    }
    
    // 控制台开关
    void enableConsole(bool enable) {
        consoleEnabled = enable;
    }
    
    // 是否启用
    bool isFileEnabled() const { return fileEnabled; }
    bool isConsoleEnabled() const { return consoleEnabled; }
    
    // 核心日志函数
    void log(const string& msg, bool forceConsole = false) {
        // 控制台输出（除非强制隐藏）
        if (consoleEnabled || forceConsole) {
            cout << msg << endl;
        }
        
        // 文件输出
        if (!fileEnabled) return;
        
        lock_guard<mutex> lock(logMtx);
        if (logFile.is_open()) {
            auto now = chrono::system_clock::now();
            auto time_t = chrono::system_clock::to_time_t(now);
            
            char timeBuf[32];
            strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", localtime(&time_t));
            
            logFile << "[" << timeBuf << "] " << msg << endl;
            logFile.flush();
        }
    }
    
    // 调试日志（受开关控制）
    void logDebug(const string& msg) {
        log("[DEBUG] " + msg);
    }
    
    // 错误日志（强制输出到控制台）
    void logError(const string& msg) {
        log("[ERROR] " + msg, true);  // forceConsole = true
    }
    
    // 信息日志
    void logInfo(const string& msg) {
        log("[INFO] " + msg);
    }
    
    // 关键信息（强制输出）
    void logImportant(const string& msg) {
        log("[重要] " + msg, true);
    }
};

// 全局日志对象
Logger logger;
// ===================== 超参 =====================
const int epochs = 3;
struct HyperParams {
    int MAX_GEN_STEP = 100;
    int VOCAB_SIZE = 131072;
    int EDGE_DECAY_STEP = 19;
    int PRUNE_WEIGHT_THRESH = 2;
    int MIN_SCORE = 1;
    int MAX_LINK_OFFSET = 3;
    int LONG_RANGE_LINK = 24;
    int SHORT_MEMORY_SIZE = 24;
    int ATTENTION_SPAN = 24;
    int REFLECT_STRENGTH = 10;
    int AMYGDALA_MEM_SIZE = 10;
    int CORTEX_LAYERS = 6;
    int PERMANENT_WEIGHT_THRESH = 50;
    int GEN_TEMP = 55;
    int TOP_K_CAND = 28;
    int SELF_CHECK_ROUND = 3;
    int CREATE_CHAR_RATE_NORMAL = 18;
    int CREATE_CHAR_RATE_EXCITE = 14;
    int MAX_DECODE_DEPTH = 10;
    int LONG_TERM_LOGIC_MEMORY = 128;
    int LOGIC_RELATION_THRESH = 4;
    int MAX_PARAGRAPH_LENGTH = 256;
    int REASONING_DEPTH = 5;
    int SAVE_EVERY_N_ROUNDS = 5;
    int FLUENCY_BASE_SCORE = 30;
    int PUNCT_DUPLICATE_PENALTY = 4;
    int CHAR_DUPLICATE_PENALTY = 3;
    int ABNORMAL_SEG_PENALTY = 5;
    int POTENTIAL_MIN = -10;
    int POTENTIAL_MAX = 30;
    int ACTIVE_THRESH = 2;
    int ENERGY_MAX = 20;
    int NEUROGENESIS_QUALITY_THRESH = 30;
    int NEUROGENESIS_COOLDOWN = 5;
    int MAX_NEURONS_PER_LAYER = 300;
    int INACTIVE_PRUNE_STEPS = 100;
    int REWIRING_INTERVAL = 20;
    int REWIRING_EDGES = 5;
    int EPISODIC_MEMORY_LIMIT = 200;
    int REASONING_CONFIDENCE_THRESH = 30;
    int PLANNING_MAX_STEPS = 15;
    int SELF_NEURON_COUNT = 20;
    int PREDICTION_ERROR_THRESH = 30;
    float SELF_PREDICTION_ALPHA = 0.9f;
    int CONSCIOUSNESS_REPORT_INTERVAL = 3;
    float CURIOSITY_BOOST = 1.5f;
    int INNER_GOAL_CHECK_INTERVAL = 5;
	// ========== 新增：预测误差相关参数 ==========
    float ERROR_LEARNING_RATE = 0.3f;        // 误差学习率
    float SURPRISE_THRESHOLD = 3.0f;         // 触发反向推理的惊喜阈值
    float ERROR_DECAY_RATE = 0.7f;           // 误差衰减率
    int MAX_ERROR_HISTORY = 20;              // 每个边保留的误差历史
    map<string, pair<int, int>> safeRanges = {
        {"MAX_GEN_STEP", {50, 300}}, {"VOCAB_SIZE", {10000, 100000}},
        {"EDGE_DECAY_STEP", {10, 50}}, {"PRUNE_WEIGHT_THRESH", {1, 5}},
        {"MIN_SCORE", {1, 10}}, {"MAX_LINK_OFFSET", {1, 8}},
        {"LONG_RANGE_LINK", {10, 50}}, {"SHORT_MEMORY_SIZE", {10, 50}},
        {"ATTENTION_SPAN", {10, 50}}, {"REFLECT_STRENGTH", {5, 20}},
        {"AMYGDALA_MEM_SIZE", {5, 30}}, {"CORTEX_LAYERS", {3, 12}},
        {"PERMANENT_WEIGHT_THRESH", {40, 80}}, {"GEN_TEMP", {20, 90}},
        {"TOP_K_CAND", {10, 50}}, {"SELF_CHECK_ROUND", {2, 8}},
        {"CREATE_CHAR_RATE_NORMAL", {10, 30}}, {"CREATE_CHAR_RATE_EXCITE", {8, 25}},
        {"MAX_DECODE_DEPTH", {5, 20}}, {"LONG_TERM_LOGIC_MEMORY", {50, 300}},
        {"LOGIC_RELATION_THRESH", {2, 10}}, {"MAX_PARAGRAPH_LENGTH", {100, 500}},
        {"REASONING_DEPTH", {2, 8}}, {"SAVE_EVERY_N_ROUNDS", {3, 15}},
        {"FLUENCY_BASE_SCORE", {20, 50}}, {"PUNCT_DUPLICATE_PENALTY", {2, 12}},
        {"CHAR_DUPLICATE_PENALTY", {2, 10}}, {"ABNORMAL_SEG_PENALTY", {2, 12}},
        {"POTENTIAL_MIN", {-20, -5}}, {"POTENTIAL_MAX", {20, 50}},
        {"ACTIVE_THRESH", {1, 5}}, {"ENERGY_MAX", {15, 40}}
    };
    map<string, deque<int>> paramHistory;
    deque<int> qualityHistory;
    int getAverageQuality() { if(qualityHistory.empty()) return 50; int s=0; for(int q:qualityHistory) s+=q; return s/qualityHistory.size(); }
    void recordQuality(int q) { qualityHistory.push_back(q); if(qualityHistory.size()>20) qualityHistory.pop_front(); }
    void autoTune(int q);
    void adjustParam(const string& name, int delta, int minVal, int maxVal);
    int* getParamPtr(const string& name);
    void reportParams();
};
HyperParams hp;

const string MODEL_FILE="model.bin";
const string DYNAMIC_TRAIN_FILE="dynamic_train.txt";
const string KNOWLEDGE_FILE="knowledge.txt";

enum NeuronMode { MODE_UNKNOWN=0, MODE_LANGUAGE=1, MODE_CONCEPT=2, MODE_LOGIC=3, MODE_ATTENTION=4, MODE_MEMORY=5, MODE_EMOTION=6, MODE_ACTION=7 };
enum EmotionType { EMO_NORMAL, EMO_EXCITE, EMO_LOW, EMO_QUEST };
enum StyleIntent { STYLE_STORY, STYLE_EMOTION, STYLE_SIMPLE };
enum LogicRelation { LOGIC_NONE=0, LOGIC_CAUSE=1, LOGIC_PROGRESS=2, LOGIC_TURN=3, LOGIC_SUMMARY=4, LOGIC_PARALLEL=5, LOGIC_CONDITION=6 };
enum AutoPosType { POS_UNKNOWN=0, POS_NOUN=1, POS_VERB=2, POS_ADJ=3, POS_AUX=4, POS_PUNCT=5, POS_PRON=6, POS_ADV=7, POS_PREP=8, POS_CONJ=9, POS_NUM=10, POS_QUANT=11, POS_INTERJ=12 };

recursive_mutex cortexMtx;
recursive_mutex tokenMtx;

atomic<bool> userInputWaiting(false);

struct pair_hash {
    template<class T1,class T2> size_t operator()(const pair<T1,T2>& p) const {
        return hash<T1>{}(p.first) ^ (hash<T2>{}(p.second)<<1);
    }
};
// 语境快照，用于区分不同上下文中的关系强度
struct ContextSnapshot {
    int prevPos;       // 前一个词的词性 (AutoPosType)
    int nextPos;       // 后一个词的词性
    int prevToken;     // 前一个词的 token ID
    int nextToken;     // 后一个词的 token ID
    int logicType;     // 逻辑关系类型 (LogicRelation)
    bool isStart;      // 是否句首
    bool isEnd;        // 是否句尾

    uint64_t hash() const {
        uint64_t h = 0;
        h ^= (uint64_t)prevPos << 0;
        h ^= (uint64_t)nextPos << 8;
        h ^= (uint64_t)prevToken << 16;
        h ^= (uint64_t)nextToken << 32;
        h ^= (uint64_t)logicType << 48;
        h ^= (uint64_t)isStart << 56;
        h ^= (uint64_t)isEnd << 57;
        return h;
    }
};
struct DynamicEdge {
	int decayAge = 0;  // 自上次强化以来的衰减步数
    int target;
    int weight;
    int permanent;
    int lifeCycle;
    LogicRelation logic;
    std::unordered_map<uint64_t, int> contextWeights;

    // ========== 新增：预测误差相关 ==========
    float predictionError = 0.0f;        // 近期预测误差累积
    int errorUpdateCount = 0;            // 更新次数
    deque<float> errorHistory;           // 误差历史（用于趋势分析）

    DynamicEdge() : target(-1), weight(0), permanent(0), lifeCycle(19), logic(LOGIC_NONE) {}
    DynamicEdge(int t) : target(t), weight(0), permanent(0), lifeCycle(19), logic(LOGIC_NONE) {}

    // 更新误差
    void updateError(float error) {
        predictionError = predictionError * 0.7f + error * 0.3f;
        errorUpdateCount++;
        errorHistory.push_back(error);
        if (errorHistory.size() > 20) errorHistory.pop_front();
    }

    // 误差趋势（正=误差增大，负=误差减小）
    float errorTrend() const {
        if (errorHistory.size() < 5) return 0;
        float recent = 0, old = 0;
        int n = (int)errorHistory.size();
        for (int i = n-5; i < n; i++) recent += errorHistory[i];
        for (int i = 0; i < 5; i++) old += errorHistory[i];
        return (recent/5.0f) - (old/5.0f);
    }

    // 添加上下文权重
    void addContext(const ContextSnapshot& snap, int delta = 1) {
        uint64_t h = snap.hash();
        contextWeights[h] += delta;
        if (contextWeights[h] > 1000) contextWeights[h] = 1000;
    }

    int getContextWeight(const ContextSnapshot& snap) const {
        auto it = contextWeights.find(snap.hash());
        return (it != contextWeights.end()) ? it->second : 0;
    }

    int total() const {
        int sum = weight + permanent;
        for (auto& kv : contextWeights) sum += kv.second / 10;
        return sum;
    }

    float importance() const {
        float imp = permanent * 2.0f + weight + lifeCycle / 10.0f;
        for (auto& kv : contextWeights) imp += kv.second / 20.0f;
        // 高误差增加重要性（需要关注）
        if (predictionError > 5.0f) imp += predictionError * 0.5f;
        return imp;
    }

	void decay() {
	    if (permanent > 0) return;
	
	    if (lifeCycle > 0) {
	        lifeCycle = (int)(lifeCycle * 0.95f);
	        if (lifeCycle < 1) lifeCycle = 0;
	        return;
	    }
	
	    decayAge++;  // 现在每个边独立计数
	    const float decayFactor = 0.03f;
	    float fWeight = (float)weight;
	    fWeight = fWeight / (1.0f + decayAge * decayFactor);
	    weight = (int)fWeight;
	    if (weight < 1) weight = 0;
	}
	
	void boost() {
	    weight += 10;
	    lifeCycle = hp.EDGE_DECAY_STEP;  // 重置宽限期
	    decayAge = 0; 
	    // 永久化门槛降低，且永久化后不再腰斩
	    if (weight >= hp.PERMANENT_WEIGHT_THRESH) {
	        permanent += 1;
	        // 删除 weight /= 2;  // 不再腰斩！
	    }
	}
	
	void boostStrong(int val) {
	    weight += val;
	    lifeCycle = hp.EDGE_DECAY_STEP;
	    decayAge = 0; 
	    // 永久化：每满50点权重转1点永久，保留剩余权重
	    if (weight >= hp.PERMANENT_WEIGHT_THRESH) {
	        int newPermanent = weight / hp.PERMANENT_WEIGHT_THRESH;
	        permanent += newPermanent;
	        weight = weight % hp.PERMANENT_WEIGHT_THRESH;  // 保留余数
	        // 如果余数太小，给个保底
	        if (weight < 5 && permanent > 0) weight += 5;
	    }
	}
};

struct ReasoningTrace {
    int token;
    LogicRelation relation;
    float confidence;
    vector<int> supportingTokens;
};

struct Episode {
    vector<int> tokens;
    vector<float> embedding;
    chrono::system_clock::time_point timestamp;
    float importance;
};

struct GlobalWorkspace {
    vector<pair<int, float>> content;
    mutex mtx;
    void broadcast(int token, float salience) {
        lock_guard<mutex> lock(mtx);
        content.push_back({token, salience});
        if (content.size() > 7) content.erase(content.begin());
    }
    vector<int> getTop(int k) {
        lock_guard<mutex> lock(mtx);
        sort(content.begin(), content.end(), [](const pair<int, float>& a, const pair<int, float>& b) { return a.second > b.second; });
        vector<int> res;
        for (int i = 0; i < min(k, (int)content.size()); i++) res.push_back(content[i].first);
        return res;
    }
    void clear() {
        lock_guard<mutex> lock(mtx);
        content.clear();
    }
};
// ========== 预测误差相关 ==========
struct PredictionRecord {
    int predictedToken;
    int actualToken;
    float predictionScore;
    float surprise;
    vector<int> context;
    chrono::system_clock::time_point timestamp;
};
struct Neuron {
    // 基础属性
    int neuronId = 0;
    int layer = 0;
    int potential = 0;
    int activation = 0;
    int inhibition = 0;
    int energy = 0;
    int maturity = 0;
    int emotionEnergy = 0;
    AutoPosType autoPos = POS_UNKNOWN;
    int posScore[13] = {0};
    NeuronMode mode = MODE_UNKNOWN;
    int modeStrength = 0;
    vector<NeuronMode> compatibleModes;
    unordered_set<int> boundTokens;
    vector<pair<int, int>> tokenScoreVec;

    // 输入/输出边
    vector<DynamicEdge> inputs;
    vector<DynamicEdge> outputs;

    // 传播相关
    int layerFeature = 0;
    int inactiveSteps = 0;

    // 推理记录
    vector<ReasoningTrace> reasoningLinks;
    unordered_map<int, LogicRelation> tokenLogicMap;

    // ---------- 语境桶：语境ID → 目标token列表 ----------
    vector<vector<int>> contextBuckets;

    // ---------- 新增：位置轨迹 ----------
    // 语境ID → 该token在该语境下出现的位置百分比列表
    unordered_map<int, vector<int>> contextPositionPcts;

    // ---------- 构造函数 ----------
    Neuron() = default;
    Neuron(int nid, int l) : neuronId(nid), layer(l) {}
    // ---------- 获取带匹配度的边列表 ----------
	std::vector<std::pair<int, float>> getEdgesWithMatchScore(int contextID) const;

    // ---------- 向指定语境添加目标token ----------
    void addTargetToContext(int contextID, int targetToken);
    // ---------- 新增：记录位置 ----------
    void recordPosition(int contextID, int positionPercent) {
        if (contextID < 0) return;
        // 直接插入，允许重复（保留每次出现的具体位置）
        contextPositionPcts[contextID].push_back(positionPercent);
        // 限制列表长度，防止无限增长
        if (contextPositionPcts[contextID].size() > 200) {
            contextPositionPcts[contextID].erase(
                contextPositionPcts[contextID].begin(),
                contextPositionPcts[contextID].begin() + 50
            );
        }
    }

    // ---------- 新增：获取位置数据 ----------
    const vector<int>* getPositionsForContext(int contextID) const {
        auto it = contextPositionPcts.find(contextID);
        if (it != contextPositionPcts.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    // ---------- 新增：计算位置匹配得分 ----------
    int calcPositionMatchScore(int contextID, int currentPosPct) const {
        auto it = contextPositionPcts.find(contextID);
        if (it == contextPositionPcts.end()) return 0;
        const vector<int>& positions = it->second;
        if (positions.empty()) return 0;

        int score = 0;
        for (int histPos : positions) {
            int dist = abs(histPos - currentPosPct);
            if (dist < 3) score += 25;
            else if (dist < 8) score += 15;
            else if (dist < 15) score += 8;
            else if (dist < 25) score += 3;
        }
        return score;
    }

    // ---------- 原有成员函数（声明） ----------
    void setMode(NeuronMode m, int strength = 5);
    bool canConnect(const Neuron& o) const;
    void integrate(int sig);
    void activate();
    void bindToken(int tid, int sc = 1);
    int getBestToken() const;
    DynamicEdge* findOutput(int tid);
    DynamicEdge* findInput(int tid);
    void linkOut(int tid, LogicRelation logic = LOGIC_NONE);
    void linkIn(int tid);
    void addReasoning(int cid, LogicRelation l, float conf, const vector<int>& ctx);
    void pruneWeak();
    void updateEdges();

    // ---------- 新增：清空位置数据（用于重置或调试） ----------
    void clearPositionData() {
        contextPositionPcts.clear();
    }
    // ========== 新增：预测误差相关 ==========
    float predictionConfidence = 0.5f;    // 预测置信度
    float errorAccumulator = 0.0f;        // 累积误差（触发重评）
    unordered_map<uint64_t, int> contextExpectations;  // 语境→期望token映射
    vector<PredictionRecord> recentPredictions;        // 近期预测记录
    // ========== 新增：语境期望管理 ==========
    void addContextExpectation(const ContextSnapshot& snap, int expectedToken) {
        uint64_t h = snap.hash();
        contextExpectations[h] = expectedToken;
        if (contextExpectations.size() > 100) {
            // 保留最近的50个
            auto it = contextExpectations.begin();
            advance(it, 50);
            contextExpectations.erase(it, contextExpectations.end());
        }
    }
    
    int getContextExpectation(const ContextSnapshot& snap) const {
        auto it = contextExpectations.find(snap.hash());
        return (it != contextExpectations.end()) ? it->second : -1;
    }
};

struct EmotionState { EmotionType type=EMO_NORMAL; int intensity=0; };

// ===================== WorldSimulator - 完全基于 token ID =====================
class WorldSimulator {
private:
    struct DialogueTurn {
        vector<int> userInput;
        vector<int> systemOutput;
        bool feedbackPositive;
        int turnIndex;
    };

    vector<DialogueTurn> history;
    unordered_map<int, float> userPrefVector;   // token → 偏好强度（频率累积）
    unordered_set<int> userPositiveTokens;      // 用户喜欢的 token（从正反馈中挖掘）
    unordered_set<int> userNegativeTokens;      // 用户不喜欢的 token（从负反馈中挖掘）

    float avgTurnLength = 0.0f;
    int totalTurns = 0;
    int positiveFeedbackCount = 0;
    int negativeFeedbackCount = 0;

public:
    // 记录一次对话，反馈已知
    void recordTurn(const vector<int>& userInput, const vector<int>& systemOutput, bool feedback) {
        history.push_back({userInput, systemOutput, feedback, (int)history.size()});
        totalTurns++;
        if (feedback) positiveFeedbackCount++;
        else negativeFeedbackCount++;

        // 更新偏好向量（词频累积，带衰减）
        for (int tid : userInput) {
            userPrefVector[tid] += 1.0f;
        }
        // 全局衰减，让最近偏好更显著
        for (auto& kv : userPrefVector) {
            kv.second *= 0.99f;
        }

        // 更新平均输入长度
        avgTurnLength = (avgTurnLength * (totalTurns - 1) + userInput.size()) / totalTurns;

        // 根据反馈标记用户喜欢的/不喜欢的 token（挖掘情感）
        if (feedback) {
            // 正反馈：用户输入中的 token 视为正面
            for (int tid : userInput) {
                userPositiveTokens.insert(tid);
                // 同时从负面集中移除（避免冲突）
                userNegativeTokens.erase(tid);
            }
        } else {
            // 负反馈：用户输入中的 token 视为负面
            for (int tid : userInput) {
                userNegativeTokens.insert(tid);
                userPositiveTokens.erase(tid);
            }
        }
    }

    // 预测用户对给定候选输出的满意度（0~1）
    float predictSatisfaction(const vector<int>& candidateOutput, const vector<int>& context) {
        if (totalTurns < 2) return 0.5f;

        float score = 0.5f;

        // 1. 偏好重合度（候选输出中 token 在偏好向量中的加权平均）
        float overlap = 0.0f;
        int candLen = candidateOutput.size();
        if (candLen > 0) {
            for (int tid : candidateOutput) {
                auto it = userPrefVector.find(tid);
                if (it != userPrefVector.end()) overlap += it->second;
            }
            overlap /= candLen;
            overlap = min(1.0f, overlap / 5.0f); // 假设最大平均偏好为 5
            score = 0.5f + 0.3f * (overlap - 0.5f);
        }

        // 2. 长度匹配度
        float lenRatio = (float)candidateOutput.size() / (avgTurnLength + 1);
        float lenMatch = 1.0f - fabs(lenRatio - 1.0f) * 0.5f;
        lenMatch = max(0.0f, min(1.0f, lenMatch));
        score = 0.6f * score + 0.4f * lenMatch;

        // 3. 情感一致性：检查候选输出中是否包含用户喜欢/讨厌的 token
        int posHit = 0, negHit = 0;
        for (int tid : candidateOutput) {
            if (userPositiveTokens.count(tid)) posHit++;
            if (userNegativeTokens.count(tid)) negHit++;
        }
        float posRatio = min(1.0f, posHit * 0.2f);   // 每出现一个正面词加 0.2，上限 1
        float negRatio = min(1.0f, negHit * 0.3f);   // 负面词惩罚更重
        float emotScore = 1.0f - negRatio + 0.5f * posRatio;
        emotScore = max(0.0f, min(1.0f, emotScore));
        score = 0.7f * score + 0.3f * emotScore;

        // 4. 历史反馈倾向
        float feedbackBias = (float)positiveFeedbackCount / (totalTurns + 1);
        score = 0.8f * score + 0.2f * feedbackBias;

        return max(0.0f, min(1.0f, score));
    }

    // 反事实预演：对候选 token 列表，返回 (token, 满意度分数)
    vector<pair<int, float>> simulateCandidates(const vector<int>& candidateTokens,
                                                const vector<int>& context) {
        vector<pair<int, float>> results;
        for (int tid : candidateTokens) {
            vector<int> fullOutput = context;
            fullOutput.push_back(tid);
            float sat = predictSatisfaction(fullOutput, context);
            results.emplace_back(tid, sat);
        }
        return results;
    }

    bool hasData() const { return totalTurns > 1; }
    string getSummary() const {
        stringstream ss;
        ss << "总轮次:" << totalTurns
           << " 积极率:" << (totalTurns ? (100 * positiveFeedbackCount / totalTurns) : 0) << "%"
           << " 平均长度:" << avgTurnLength
           << " 偏好词数:" << userPrefVector.size()
           << " 正面token数:" << userPositiveTokens.size()
           << " 负面token数:" << userNegativeTokens.size();
        return ss.str();
    }

    // 保留原有占位函数（兼容性）
    void step(int action) {}
    vector<float> getObservation() { return {}; }
    int getStateHash() { return 0; }
};

// 全局用户模型（类似 Logger）
WorldSimulator userModel;
// ===================== GBK ? UTF-8 编码转换 =====================

// GBK → UTF-8
string gbkToUtf8(const string& gbk) {
    if (gbk.empty()) return "";

    // 1. GBK → UTF-16 (宽字符)
    int lenW = MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, NULL, 0);
    if (lenW == 0) return "";
    wstring wstr(lenW, L'\0');
    MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, &wstr[0], lenW);
    // 去掉末尾的 '\0'
    while (!wstr.empty() && wstr.back() == L'\0') wstr.pop_back();

    // 2. UTF-16 → UTF-8
    int lenU8 = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (lenU8 == 0) return "";
    string utf8(lenU8, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8[0], lenU8, NULL, NULL);
    while (!utf8.empty() && utf8.back() == '\0') utf8.pop_back();

    return utf8;
}

// UTF-8 → GBK
string utf8ToGbk(const string& utf8) {
    if (utf8.empty()) return "";

    // 1. UTF-8 → UTF-16
    int lenW = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, NULL, 0);
    if (lenW == 0) return "";
    wstring wstr(lenW, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wstr[0], lenW);
    while (!wstr.empty() && wstr.back() == L'\0') wstr.pop_back();

    // 2. UTF-16 → GBK
    int lenGBK = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (lenGBK == 0) return "";
    string gbk(lenGBK, '\0');
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &gbk[0], lenGBK, NULL, NULL);
    while (!gbk.empty() && gbk.back() == '\0') gbk.pop_back();

    return gbk;
}

// ============================================================
//  WordSegmenter - 按词切分，GBK 编码
// ============================================================

class WordSegmenter {
private:
    // 词频词典：词 → 词频
    std::unordered_map<std::string, int> dict;
    int maxWordLen = 0;  // 词典中最长词的长度（按字符数）
    bool loaded = false;

    bool loadDict(const string& dictPath) {
	    ifstream f(dictPath);
	    if (!f.is_open()) {
	        cerr << "[分词] 无法打开词典: " << dictPath << endl;
	        return false;
	    }
	
	    // 先统计词条数
	    int lineCount = 0;
	    string line;
	    while (getline(f, line)) {
	        if (!line.empty()) lineCount++;
	    }
	    f.clear();
	    f.seekg(0, ios::beg);
	
	    // 预分配空间，避免反复 rehash
	    dict.reserve(lineCount);
	    dict.max_load_factor(0.7);  // 降低负载因子，减少冲突
	
	    maxWordLen = 0;
	    while (getline(f, line)) {
	        if (line.empty()) continue;
	        size_t space1 = line.find(' ');
	        if (space1 == string::npos) continue;
	        size_t space2 = line.find(' ', space1 + 1);
	
	        string utf8Word = line.substr(0, space1);
	        if (utf8Word.empty()) continue;
	
	        // UTF-8 → GBK
	        string gbkWord = utf8ToGbk(utf8Word);
	
	        int freq = 1;
	        if (space2 != string::npos) {
	            string freqStr = line.substr(space1 + 1, space2 - space1 - 1);
	            freq = atoi(freqStr.c_str());
	            if (freq <= 0) freq = 1;
	        }
	
	        dict[gbkWord] = freq;
	        int byteLen = gbkWord.size();
	        if (byteLen > maxWordLen) maxWordLen = byteLen;
	    }
	
	    loaded = true;
	    cout << "[分词] 加载完成，词条数: " << dict.size() << "，最大词长: " << maxWordLen << " 字节" << endl;
	    return true;
	}

    // 检查一个词是否在词典中
    bool inDict(const std::string& word) const {
        return dict.find(word) != dict.end();
    }

    // 正向最大匹配（FMM）
    std::vector<std::string> fmm(const std::string& text) const {
        std::vector<std::string> result;
        int pos = 0;
        int n = text.size();

        while (pos < n) {
            int len = std::min(maxWordLen, n - pos);
            std::string word = text.substr(pos, len);

            // 如果整段不在词典中，尝试缩短
            while (len > 1 && !inDict(word)) {
                // 检查是否是单字词（如果是单字，直接切）
                if (len == 1) break;
                // 检查前一个字符是否在 GBK 范围内
                len -= 1;
                // 如果 len 指向 GBK 双字节的中间，跳过
                while (len > 0 && ((unsigned char)text[pos + len - 1] >= 0x81 && (unsigned char)text[pos + len - 1] <= 0xFE)) {
                    len -= 1;
                }
                word = text.substr(pos, len);
            }

            // 如果还是不在词典中，按单字切分（但要正确处理 GBK 双字节）
            if (!inDict(word)) {
                // 检查是否是 GBK 双字节字符
                unsigned char c = (unsigned char)text[pos];
                if (c >= 0x81 && c <= 0xFE) {
                    // 双字节
                    if (pos + 1 < n) {
                        word = text.substr(pos, 2);
                        result.push_back(word);
                        pos += 2;
                    } else {
                        // 单字节
                        word = text.substr(pos, 1);
                        result.push_back(word);
                        pos += 1;
                    }
                } else {
                    // 单字节 ASCII
                    word = text.substr(pos, 1);
                    result.push_back(word);
                    pos += 1;
                }
            } else {
                result.push_back(word);
                pos += word.size();
            }
        }

        return result;
    }

    // 反向最大匹配（BMM）
    std::vector<std::string> bmm(const std::string& text) const {
        std::vector<std::string> result;
        int pos = text.size();

        while (pos > 0) {
            int len = std::min(maxWordLen, pos);
            std::string word = text.substr(pos - len, len);

            while (len > 1 && !inDict(word)) {
                if (len == 1) break;
                len -= 1;
                // 避免截断 GBK 双字节
                while (len > 0 && ((unsigned char)text[pos - len] >= 0x81 && (unsigned char)text[pos - len] <= 0xFE)) {
                    len -= 1;
                }
                if (len <= 0) break;
                word = text.substr(pos - len, len);
            }

            if (!inDict(word)) {
                // 按单字处理
                unsigned char c = (unsigned char)text[pos - 1];
                if (c >= 0x81 && c <= 0xFE) {
                    if (pos - 2 >= 0) {
                        word = text.substr(pos - 2, 2);
                        result.insert(result.begin(), word);
                        pos -= 2;
                    } else {
                        word = text.substr(pos - 1, 1);
                        result.insert(result.begin(), word);
                        pos -= 1;
                    }
                } else {
                    word = text.substr(pos - 1, 1);
                    result.insert(result.begin(), word);
                    pos -= 1;
                }
            } else {
                result.insert(result.begin(), word);
                pos -= word.size();
            }
        }

        return result;
    }

public:
    WordSegmenter() {}

    // 加载词典
    bool load(const std::string& dictPath) {
        return loadDict(dictPath);
    }

    bool isLoaded() const { return loaded; }

    // 分词主函数（双向最大匹配，取更优结果）
    std::vector<std::string> cut(const std::string& text) {
        if (!loaded) {
            std::cerr << "[错误] 词典尚未加载" << std::endl;
            return {};
        }

        if (text.empty()) return {};

        // 正向最大匹配
        std::vector<std::string> fmmResult = fmm(text);

        // 反向最大匹配
        std::vector<std::string> bmmResult = bmm(text);

        // 选择结果更优的（词数更少，词长更长）
        if (fmmResult.size() <= bmmResult.size()) {
            return fmmResult;
        } else {
            return bmmResult;
        }
    }

    // 获取词典大小
    size_t dictSize() const { return dict.size(); }
    int getMaxWordLen() const { return maxWordLen; }
};
/*
// ============================================================
//  PosDictLoader - 加载 pos_dict 文件，提供词性查询
// ============================================================

class PosDictLoader {
private:
    // 字 → vector<{状态, 词性}>
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> dict;
    bool loaded = false;

    // 状态映射到 AutoPosType
    AutoPosType mapStateToPos(const std::string& state) {
        if (state == "v" || state == "vd" || state == "vi" || state == "vl" || state == "vg") return POS_VERB;
        if (state == "n" || state == "nr" || state == "ns" || state == "nt" || state == "nz" || state == "vn") return POS_NOUN;
        if (state == "a" || state == "ad" || state == "an") return POS_ADJ;
        if (state == "r" || state == "rr" || state == "rz") return POS_PRON;
        if (state == "p") return POS_PREP;
        if (state == "c") return POS_CONJ;
        if (state == "u" || state == "ul" || state == "uz") return POS_AUX;
        if (state == "m") return POS_NUM;
        if (state == "q") return POS_QUANT;
        if (state == "d") return POS_ADV;
        if (state == "x" || state == "w" || state == "wp" || state == "ws" || state == "wu") return POS_PUNCT;
        if (state == "e") return POS_INTERJ;
        return POS_UNKNOWN;
    }

public:
    // 加载 pos_dict 文件（GBK 编码）
    bool load(const string& filename) {
	    ifstream f(filename);
	    if (!f.is_open()) {
	        cerr << "[PosDict] 无法打开文件: " << filename << endl;
	        return false;
	    }
	
	    dict.clear();
	    string line;
	
	    while (getline(f, line)) {
	        if (line.empty() || line[0] == '#') continue;
	
	        size_t colon = line.find(':');
	        if (colon == string::npos) continue;
	
	        string word = line.substr(0, colon);
	        string rest = line.substr(colon + 1);
	
	        vector<pair<string, string>> states;
	
	        size_t pos = 0;
	        while (pos < rest.size()) {
	            size_t semicolon = rest.find(';', pos);
	            if (semicolon == string::npos) break;
	
	            string state = rest.substr(pos, semicolon - pos);
	            if (!state.empty()) {
	                size_t comma = state.find(',');
	                if (comma != string::npos) {
	                    string tag = state.substr(0, comma);
	                    string posTag = state.substr(comma + 1);
	                    // 去空格
	                    tag.erase(0, tag.find_first_not_of(" \t"));
	                    tag.erase(tag.find_last_not_of(" \t") + 1);
	                    posTag.erase(0, posTag.find_first_not_of(" \t"));
	                    posTag.erase(posTag.find_last_not_of(" \t") + 1);
	                    states.push_back({tag, posTag});
	                }
	            }
	            pos = semicolon + 1;
	        }
	
	        if (!states.empty()) {
	            dict[word] = states;
	        }
	    }
	
	    loaded = true;
	    cout << "[PosDict] 加载完成，共 " << dict.size() << " 个条目" << endl;
	    return true;
	}

    // 查询一个词的词性（取第一个出现的词性）
    AutoPosType getPos(const std::string& word) {
        auto it = dict.find(word);
        if (it == dict.end()) return POS_UNKNOWN;

        const auto& states = it->second;
        if (states.empty()) return POS_UNKNOWN;

        // 优先取 E 或 S 状态（更可靠）
        for (const auto& state : states) {
            const std::string& tag = state.first;
            if (tag == "E" || tag == "S") {
                return mapStateToPos(state.second);
            }
        }

        return mapStateToPos(states[0].second);
    }

    bool isLoaded() const { return loaded; }
};
*/


// ============================================================
//  PosTaggerWrapper - 外部词性标注包装器（动态加载版）
// ============================================================

class PosTaggerWrapper {
private:
    cppjieba::Jieba* jieba = nullptr;  // 指针，延迟初始化
    std::unordered_map<std::string, AutoPosType> tagMap;
    bool initialized = false;

    void buildTagMap() {
        // 名词系列
        tagMap["n"]  = POS_NOUN;
        tagMap["nr"] = POS_NOUN;
        tagMap["ns"] = POS_NOUN;
        tagMap["nt"] = POS_NOUN;
        tagMap["nz"] = POS_NOUN;
        tagMap["vn"] = POS_NOUN;

        // 动词系列
        tagMap["v"]  = POS_VERB;
        tagMap["vd"] = POS_VERB;
        tagMap["vi"] = POS_VERB;
        tagMap["vl"] = POS_VERB;
        tagMap["vg"] = POS_VERB;

        // 形容词系列
        tagMap["a"]  = POS_ADJ;
        tagMap["ad"] = POS_ADJ;

        // 代词
        tagMap["r"]  = POS_PRON;
        tagMap["rr"] = POS_PRON;
        tagMap["rz"] = POS_PRON;

        // 介词
        tagMap["p"]  = POS_PREP;

        // 连词
        tagMap["c"]  = POS_CONJ;

        // 助词
        tagMap["u"]  = POS_AUX;
        tagMap["ul"] = POS_AUX;
        tagMap["uz"] = POS_AUX;

        // 数词
        tagMap["m"]  = POS_NUM;

        // 量词
        tagMap["q"]  = POS_QUANT;

        // 副词
        tagMap["d"]  = POS_ADV;

        // 标点
        tagMap["x"]  = POS_PUNCT;
        tagMap["w"]  = POS_PUNCT;
        tagMap["wp"] = POS_PUNCT;
        tagMap["ws"] = POS_PUNCT;
        tagMap["wu"] = POS_PUNCT;

        // 叹词
        tagMap["e"]  = POS_INTERJ;

        // 未知
        tagMap["unk"] = POS_UNKNOWN;
    }

    AutoPosType mapTagToEnum(const std::string& tag) const {
        auto it = tagMap.find(tag);
        if (it != tagMap.end()) return it->second;
        // 模糊匹配：如果标签以已知前缀开头
        for (const auto& entry : tagMap) {
            if (tag.find(entry.first) == 0) return entry.second;
        }
        return POS_UNKNOWN;
    }

    // 初始化 Jieba（只在第一次调用时执行）
    void ensureInitialized() {
        if (initialized) return;
        
        std::cout << "[Jieba] 正在加载词典..." << std::endl;
        
        // 动态创建 Jieba 对象
        jieba = new cppjieba::Jieba(
            "cppjieba/dict/jieba.dict.utf8",
            "cppjieba/dict/hmm_model.utf8",
            "cppjieba/dict/user.dict.utf8",
            "cppjieba/dict/idf.utf8",
            "cppjieba/dict/stop_words.utf8"
        );
        
        buildTagMap();
        initialized = true;
        
        std::cout << "[Jieba] 加载完成" << std::endl;
    }

public:
    PosTaggerWrapper() : jieba(nullptr), initialized(false) {}
    
    ~PosTaggerWrapper() {
        if (jieba != nullptr) {
            delete jieba;
            jieba = nullptr;
        }
    }

    // 对一段文本进行分词并标注词性（GBK 输入）
    std::vector<std::pair<std::string, std::string>> tag(const std::string& text_gbk) {
        std::vector<std::pair<std::string, std::string>> results_gbk;

        if (text_gbk.empty()) return results_gbk;

        // 懒加载：第一次调用时才初始化
        ensureInitialized();

        if (jieba == nullptr) return results_gbk;

        // GBK → UTF-8
        std::string text_utf8 = gbkToUtf8(text_gbk);
        if (text_utf8.empty()) return results_gbk;

        // 调用 Jieba 分词
        std::vector<std::pair<std::string, std::string>> results_utf8;
        jieba->Tag(text_utf8, results_utf8);

        // 将结果中的词从 UTF-8 转回 GBK
        for (const auto& pair : results_utf8) {
            std::string word_gbk = utf8ToGbk(pair.first);
            results_gbk.push_back({word_gbk, pair.second});
        }

        return results_gbk;
    }

    // 获取整段文本的 token → 词性映射表
    std::unordered_map<std::string, AutoPosType> getPosMap(const std::string& text_gbk) {
        std::unordered_map<std::string, AutoPosType> posMap;
        auto tagged = tag(text_gbk);
        for (const auto& pair : tagged) {
            posMap[pair.first] = mapTagToEnum(pair.second);
        }
        return posMap;
    }

    // 检查是否已经初始化
    bool isInitialized() const { return initialized; }
};

// ===================== TextTokenizer =====================
// ===================== TextTokenizer =====================
struct TextTokenizer {
    // ---------- 原有成员 ----------
    vector<string> vocabList;
    vector<vector<int>> conceptSeq;
    vector<vector<int>> conceptContent;
    vector<pair<string,vector<int>>> knowledgeVec;
    unordered_set<string> radicalPool;
    unordered_set<string> symbolFilter={"。","，","？","！"," ","\n","\r","、","：","；"};
    unordered_set<string> numSet={"0","1","2","3","4","5","6","7","8","9","百","千","万","亿"};
    int nextConceptId=10000;
    unordered_map<int,vector<float>> tokenEmbedding;
    unordered_map<pair<int,int>,float,pair_hash> pmiMatrix;
    unordered_map<int,unordered_map<int,int>> transCount;
    unordered_map<int,int> tokenTotalCount;
    int totalTransCount=0;
    const int EMBED_DIM=16;
    vector<AutoPosType> tokenPosCache;
    vector<bool> tokenIsPunctCache;
    vector<bool> tokenIsNewCharCache;

    vector<pair<vector<int>, int>> phraseTemplates;
    vector<pair<uint64_t, int>> posSkeletonTemplates;
    unordered_map<int, vector<int>> phrasePrefix1;
    unordered_map<uint64_t, vector<int>> phrasePrefix2;
	// ========== 新增：外部词性标注器 ==========
	PosTaggerWrapper posTagger;
	
	// ========== 新增：词性缓存（token ID → 词性枚举） ==========
	unordered_map<int, AutoPosType> posCache;
	
	// ========== 新增：缓存状态 ==========
	bool posCacheValid = false;
	string cachedText;
    // ---------- 静态函数 ----------
    static uint64_t encodePosSeq(const vector<AutoPosType>& seq) {
        uint64_t code = 0;
        for (size_t i = 0; i < seq.size() && i < 6; ++i) {
            code = (code << 8) | (uint8_t)seq[i];
        }
        return code;
    }
    static vector<AutoPosType> decodePosSeq(uint64_t code) {
        vector<AutoPosType> res;
        for (int i = 0; i < 6 && code; ++i) {
            res.push_back((AutoPosType)(code & 0xFF));
            code >>= 8;
        }
        reverse(res.begin(), res.end());
        return res;
    }

    // ========== 新增：分析一段文本，填充词性缓存 ==========
	void analyzeText(const string& text) {
	    if (text.empty()) {
	        posCache.clear();
	        posCacheValid = false;
	        return;
	    }
	
	    // 如果文本没变，复用缓存
	    if (posCacheValid && cachedText == text) {
	        return;
	    }
	
	    // 调用外部库进行词性标注
	    auto posMap = posTagger.getPosMap(text);
	
	    // 清空缓存并重新填充
	    posCache.clear();
	    for (const auto& pair : posMap) {
	        const string& word = pair.first;
	        AutoPosType pos = pair.second;
	        int tid = getTokenId(word);
	        if (tid != -1) {
	            posCache[tid] = pos;
	        }
	    }
	
	    posCacheValid = true;
	    cachedText = text;
	}
	// ========== 新增：安全的编码函数（带递归保护） ==========
    vector<int> encode_direct(const string& text) {
        vector<int> seq{1};
        size_t i = 0;
        while(i < text.size()) {
            unsigned char c = (unsigned char)text[i];
            
            if(c < 0x80) {
                string ch = text.substr(i, 1);
                int tid = getTokenId(ch);
                if(tid == -1 && (int)vocabList.size() < hp.VOCAB_SIZE) {
                    vocabList.push_back(ch);
                    tid = vocabList.size() - 1;
                    if(tokenEmbedding.find(tid) == tokenEmbedding.end()) 
                        tokenEmbedding[tid] = vector<float>(EMBED_DIM, 0.01f);
                    tokenPosCache.push_back(POS_UNKNOWN);
                    tokenIsPunctCache.push_back(symbolFilter.count(ch));
                    tokenIsNewCharCache.push_back(isNewChar(ch));
                }
                if(tid != -1) seq.push_back(tid);
                i++;
            } 
            else if(c >= 0x81 && c <= 0xFE && i + 1 < text.size()) {
                string ch = text.substr(i, 2);
                int tid = getTokenId(ch);
                if(tid == -1 && (int)vocabList.size() < hp.VOCAB_SIZE) {
                    vocabList.push_back(ch);
                    tid = vocabList.size() - 1;
                    if(tokenEmbedding.find(tid) == tokenEmbedding.end()) 
                        tokenEmbedding[tid] = vector<float>(EMBED_DIM, 0.01f);
                    tokenPosCache.push_back(POS_UNKNOWN);
                    tokenIsPunctCache.push_back(symbolFilter.count(ch));
                    tokenIsNewCharCache.push_back(isNewChar(ch));
                }
                if(tid != -1) seq.push_back(tid);
                i += 2;
            } 
            else {
                i++;
            }
        }
        return seq;
    }

    // ============================================================
    //  新增：直接创建概念（不遍历 knowledgeVec）
    // ============================================================
    int createConceptDirect(const vector<int>& tokens) {
        // 检查是否已存在（只遍历 conceptSeq）
        for(int i = 0; i < (int)conceptSeq.size(); ++i) {
            if(conceptSeq[i] == tokens) {
                return i + nextConceptId;
            }
        }
        // 创建新概念
        conceptSeq.push_back(tokens);
        conceptContent.emplace_back();
        return nextConceptId + (int)conceptSeq.size() - 1;
    }

    // ============================================================
    //  替换：loadKnowledge（安全版本）
    // ============================================================
    void loadKnowledge() {
        cout << "[知识库] 开始加载..." << endl;
        cout.flush();
        
        ifstream f(KNOWLEDGE_FILE);
        if(!f.is_open()) {
            cout << "[知识库] 文件不存在，跳过" << endl;
            return;
        }
        
        // 第一步：读取所有条目
        vector<pair<string, string>> entries;
        string line_;
        int lineNum = 0;
        
        while(getline(f, line_)) {
            lineNum++;
            if(line_.empty()) continue;
            string line = utf8ToGbk(line_);
            // 去除首尾空白
            size_t start = line.find_first_not_of(" \t\r\n");
            if(start == string::npos) continue;
            line = line.substr(start);
            size_t end = line.find_last_not_of(" \t\r\n");
            if(end != string::npos) line = line.substr(0, end + 1);
            
            // 查找分隔符（支持英文冒号和中文冒号）
            size_t col1 = line.find(':');
            size_t col2 = line.find("：");
            size_t col = (col1 != string::npos) ? col1 : col2;
            
            if(col == string::npos) {
                cout << "[知识库] 警告：第 " << lineNum << " 行格式错误，跳过" << endl;
                continue;
            }
            
            string key = line.substr(0, col);
            string val = line.substr(col + 1);
            
            // 去除键值的首尾空白
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            val.erase(val.find_last_not_of(" \t\r\n") + 1);
            
            if(key.empty() || val.empty()) {
                cout << "[知识库] 警告：第 " << lineNum << " 行键或值为空，跳过" << endl;
                continue;
            }
            
            entries.emplace_back(key, val);
        }
        f.close();
        
        cout << "[知识库] 读取完成，共 " << entries.size() << " 条知识" << endl;
        
        // 第二步：先检查哪些是新的
        vector<pair<string, string>> newEntries;
        for(auto& entry : entries) {
            bool exists = false;
            for(auto& kv : knowledgeVec) {
                if(kv.first == entry.first) {
                    exists = true;
                    break;
                }
            }
            if(!exists) {
                newEntries.push_back(entry);
            }
        }
        
        cout << "[知识库] 新增 " << newEntries.size() << " 条知识" << endl;
        
        // 第三步：先构建词表（避免在编码时修改容器）
        for(auto& entry : newEntries) {
            auto chars1 = splitGBK(entry.first);
            auto chars2 = splitGBK(entry.second);
            
            for(auto& ch : chars1) {
                if(getTokenId(ch) == -1 && (int)vocabList.size() < hp.VOCAB_SIZE) {
                    vocabList.push_back(ch);
                    int tid = vocabList.size() - 1;
                    if(tokenEmbedding.find(tid) == tokenEmbedding.end()) 
                        tokenEmbedding[tid] = vector<float>(EMBED_DIM, 0.01f);
                    tokenPosCache.push_back(POS_UNKNOWN);
                    tokenIsPunctCache.push_back(symbolFilter.count(ch));
                    tokenIsNewCharCache.push_back(isNewChar(ch));
                }
            }
            for(auto& ch : chars2) {
                if(getTokenId(ch) == -1 && (int)vocabList.size() < hp.VOCAB_SIZE) {
                    vocabList.push_back(ch);
                    int tid = vocabList.size() - 1;
                    if(tokenEmbedding.find(tid) == tokenEmbedding.end()) 
                        tokenEmbedding[tid] = vector<float>(EMBED_DIM, 0.01f);
                    tokenPosCache.push_back(POS_UNKNOWN);
                    tokenIsPunctCache.push_back(symbolFilter.count(ch));
                    tokenIsNewCharCache.push_back(isNewChar(ch));
                }
            }
        }
        
        // 第四步：安全编码和添加（现在不会在编码时修改容器）
        for(auto& entry : newEntries) {
            vector<int> ktok = encode_direct(entry.first);
            vector<int> vtok = encode_direct(entry.second);
            
            if(ktok.empty() || vtok.empty()) {
                cout << "[知识库] 警告：编码失败，跳过: " << entry.first << endl;
                continue;
            }
            
            // 存储知识
            knowledgeVec.emplace_back(entry.first, vtok);
            
            // 创建概念
            int cid = createConceptDirect(ktok);
            if(cid != -1) {
                int idx = cid - nextConceptId;
                if(idx >= 0 && idx < (int)conceptContent.size()) {
                    conceptContent[idx] = vtok;
                }
            }
        }
        
        cout << "[知识库] 加载完成！共 " << knowledgeVec.size() << " 条知识" << endl;
        cout.flush();
    }
    // ---------- 原有成员函数 ----------
    TextTokenizer();
    int getTokenId(const string& s);
    vector<string> splitGBK(const string& s);
    string composeNewChar();
    vector<int> encode(const string& text);
    string decode(const vector<int>& seq, int depth=0);
    void buildVocab(const string& text);
    int createConcept(const vector<int>& tokens, const string& name);
    bool isStablePhrase(const vector<int>& gram);
    AutoPosType getPosForToken(int tid) const;
    void setPosForToken(int tid, AutoPosType pos);
    vector<int> extractConceptsFromSeq(const vector<int>& seq);
    vector<int> matchKnowledgeConcepts(const string& input);
    vector<int> getKnowledgeContent(int conceptId);
    vector<int> splitCompoundConcepts(const string& input);
    bool isNewChar(const string& s);
    bool isPunctuation(const string& s);
    float cosineSimilarity(const vector<float>& a, const vector<float>& b);
    void updateEmbeddings();
    bool isHypothetical(int tid);
    bool isChronological(int from, int to);
    void updateTokenCache();

    // ---------- 新增：模板提取与匹配 ----------
    void extractTemplatesFromCorpus(const vector<int>& seq);
    void extractTemplatesFromExcellentFile();
    vector<int> matchPhraseByPrefix(int token1, int token2 = -1);
    AutoPosType matchPosSkeleton(const vector<AutoPosType>& recentPos);
};

// ===================== 超参数实现 =====================
void HyperParams::autoTune(int q) {
    if(q>=70) {
        adjustParam("GEN_TEMP",+2,20,90); adjustParam("CREATE_CHAR_RATE_NORMAL",-1,10,30);
        adjustParam("CREATE_CHAR_RATE_EXCITE",-1,8,25); adjustParam("TOP_K_CAND",+1,10,50);
        adjustParam("REFLECT_STRENGTH",+1,5,20); adjustParam("REASONING_DEPTH",+1,2,8);
        adjustParam("PUNCT_DUPLICATE_PENALTY",-1,2,12); adjustParam("CHAR_DUPLICATE_PENALTY",-1,2,10);
        adjustParam("ABNORMAL_SEG_PENALTY",-1,2,12);
    } else if(q>=50) {
        // neutral
    } else if(q>=30) {
        adjustParam("GEN_TEMP",-2,20,90); adjustParam("TOP_K_CAND",-1,10,50);
        adjustParam("CREATE_CHAR_RATE_NORMAL",+1,10,30); adjustParam("REFLECT_STRENGTH",-1,5,20);
        adjustParam("PUNCT_DUPLICATE_PENALTY",+1,2,12); adjustParam("CHAR_DUPLICATE_PENALTY",+1,2,10);
    } else {
        adjustParam("GEN_TEMP",-3,20,90); adjustParam("TOP_K_CAND",-2,10,50);
        adjustParam("CREATE_CHAR_RATE_NORMAL",+2,10,30); adjustParam("CREATE_CHAR_RATE_EXCITE",+1,8,25);
        adjustParam("REFLECT_STRENGTH",-2,5,20); adjustParam("REASONING_DEPTH",-1,2,8);
        adjustParam("PUNCT_DUPLICATE_PENALTY",+2,2,12); adjustParam("CHAR_DUPLICATE_PENALTY",+2,2,10);
        adjustParam("ABNORMAL_SEG_PENALTY",+1,2,12); adjustParam("ACTIVE_THRESH",-1,1,5);
        adjustParam("PRUNE_WEIGHT_THRESH",-1,1,5);
    }
    static int tuneCounter=0; if(++tuneCounter>=10) { tuneCounter=0;
        if(getAverageQuality()>60) { adjustParam("SHORT_MEMORY_SIZE",+1,10,50); adjustParam("LONG_TERM_LOGIC_MEMORY",+10,50,300); }
        else if(getAverageQuality()<40) { adjustParam("SHORT_MEMORY_SIZE",-2,10,50); adjustParam("LONG_TERM_LOGIC_MEMORY",-20,50,300); }
    }
}
void HyperParams::adjustParam(const string& name, int delta, int minVal, int maxVal) {
    int* p=getParamPtr(name); if(!p) return;
    int newVal=*p+delta; newVal=max(minVal, min(newVal, maxVal));
    paramHistory[name].push_back(newVal); if(paramHistory[name].size()>5) paramHistory[name].pop_front();
    int sum=0; for(int v:paramHistory[name]) sum+=v; int smoothed=sum/paramHistory[name].size();
    *p=smoothed;
}
int* HyperParams::getParamPtr(const string& name) {
    if(name=="MAX_GEN_STEP") return &MAX_GEN_STEP; if(name=="VOCAB_SIZE") return &VOCAB_SIZE;
    if(name=="EDGE_DECAY_STEP") return &EDGE_DECAY_STEP; if(name=="PRUNE_WEIGHT_THRESH") return &PRUNE_WEIGHT_THRESH;
    if(name=="MIN_SCORE") return &MIN_SCORE; if(name=="MAX_LINK_OFFSET") return &MAX_LINK_OFFSET;
    if(name=="LONG_RANGE_LINK") return &LONG_RANGE_LINK; if(name=="SHORT_MEMORY_SIZE") return &SHORT_MEMORY_SIZE;
    if(name=="ATTENTION_SPAN") return &ATTENTION_SPAN; if(name=="REFLECT_STRENGTH") return &REFLECT_STRENGTH;
    if(name=="AMYGDALA_MEM_SIZE") return &AMYGDALA_MEM_SIZE; if(name=="CORTEX_LAYERS") return &CORTEX_LAYERS;
    if(name=="PERMANENT_WEIGHT_THRESH") return &PERMANENT_WEIGHT_THRESH; if(name=="GEN_TEMP") return &GEN_TEMP;
    if(name=="TOP_K_CAND") return &TOP_K_CAND; if(name=="SELF_CHECK_ROUND") return &SELF_CHECK_ROUND;
    if(name=="CREATE_CHAR_RATE_NORMAL") return &CREATE_CHAR_RATE_NORMAL; if(name=="CREATE_CHAR_RATE_EXCITE") return &CREATE_CHAR_RATE_EXCITE;
    if(name=="MAX_DECODE_DEPTH") return &MAX_DECODE_DEPTH; if(name=="LONG_TERM_LOGIC_MEMORY") return &LONG_TERM_LOGIC_MEMORY;
    if(name=="LOGIC_RELATION_THRESH") return &LOGIC_RELATION_THRESH; if(name=="MAX_PARAGRAPH_LENGTH") return &MAX_PARAGRAPH_LENGTH;
    if(name=="REASONING_DEPTH") return &REASONING_DEPTH; if(name=="SAVE_EVERY_N_ROUNDS") return &SAVE_EVERY_N_ROUNDS;
    if(name=="FLUENCY_BASE_SCORE") return &FLUENCY_BASE_SCORE; if(name=="PUNCT_DUPLICATE_PENALTY") return &PUNCT_DUPLICATE_PENALTY;
    if(name=="CHAR_DUPLICATE_PENALTY") return &CHAR_DUPLICATE_PENALTY; if(name=="ABNORMAL_SEG_PENALTY") return &ABNORMAL_SEG_PENALTY;
    if(name=="POTENTIAL_MIN") return &POTENTIAL_MIN; if(name=="POTENTIAL_MAX") return &POTENTIAL_MAX;
    if(name=="ACTIVE_THRESH") return &ACTIVE_THRESH; if(name=="ENERGY_MAX") return &ENERGY_MAX;
    return nullptr;
}
void HyperParams::reportParams() {
    cout << "\n[当前参数] GEN_TEMP="<<GEN_TEMP<<" TOP_K="<<TOP_K_CAND<<" REFLECT="<<REFLECT_STRENGTH
         <<" CREATE_N="<<CREATE_CHAR_RATE_NORMAL<<" CREATE_E="<<CREATE_CHAR_RATE_EXCITE
         <<" PUNCT_PEN="<<PUNCT_DUPLICATE_PENALTY<<" CHAR_PEN="<<CHAR_DUPLICATE_PENALTY<<endl;
}

// ===================== Neuron 成员实现 =====================
void Neuron::setMode(NeuronMode m,int strength){
    mode=m; modeStrength=strength; compatibleModes.clear();
    if(m==MODE_LANGUAGE) compatibleModes={MODE_CONCEPT,MODE_ATTENTION};
    else if(m==MODE_CONCEPT) compatibleModes={MODE_LANGUAGE,MODE_LOGIC,MODE_MEMORY};
    else if(m==MODE_LOGIC) compatibleModes={MODE_CONCEPT,MODE_ATTENTION,MODE_EMOTION};
    else if(m==MODE_ATTENTION) compatibleModes={MODE_LANGUAGE,MODE_LOGIC,MODE_MEMORY};
    else if(m==MODE_MEMORY) compatibleModes={MODE_CONCEPT,MODE_ATTENTION,MODE_EMOTION};
    else if(m==MODE_EMOTION) compatibleModes={MODE_MEMORY,MODE_ACTION};
    else if(m==MODE_ACTION) compatibleModes={MODE_EMOTION,MODE_LANGUAGE};
}
bool Neuron::canConnect(const Neuron& o) const { for(auto m:compatibleModes) if(m==o.mode) return true; return false; }
void Neuron::integrate(int sig) { potential=max(hp.POTENTIAL_MIN, min(hp.POTENTIAL_MAX, potential+sig)); }
void Neuron::activate() {
    int thr=hp.ACTIVE_THRESH+modeStrength/3;
    if(potential>thr) {
        activation=potential+layerFeature/2;
        energy+=activation/5;
        inactiveSteps = 0;
    } else {
        activation=0;
        energy=energy*95/100;
        inactiveSteps++;
    }
    energy=max(0,min(hp.ENERGY_MAX,energy)); inhibition=max(0,inhibition-1);
}
void Neuron::bindToken(int tid,int sc){
    if(tid<=0) return;
    boundTokens.insert(tid);
    for(auto& p:tokenScoreVec) if(p.first==tid) { p.second+=sc; return; }
    tokenScoreVec.emplace_back(tid,sc);
}
int Neuron::getBestToken() const {
    if(boundTokens.empty()) return -1;
    int best=-1,ms=-1; for(auto& p:tokenScoreVec) if(p.second>ms) ms=p.second,best=p.first;
    return best;
}
DynamicEdge* Neuron::findOutput(int tid) { for(auto& e:outputs) if(e.target==tid) return &e; return nullptr; }
DynamicEdge* Neuron::findInput(int tid) { for(auto& e:inputs) if(e.target==tid) return &e; return nullptr; }
void Neuron::linkOut(int tid, LogicRelation logic){
    if(tid<0) return;
    DynamicEdge* e=findOutput(tid);
    if(!e) { outputs.emplace_back(tid); e=&outputs.back(); }
    e->logic=logic; e->boost(); maturity=min(50,maturity+1);
}
void Neuron::linkIn(int tid){
    if(tid<0) return;
    DynamicEdge* e=findInput(tid);
    if(!e) { inputs.emplace_back(tid); e=&inputs.back(); }
    e->boost();
}
void Neuron::addReasoning(int cid,LogicRelation l,float conf,const vector<int>& ctx){
    reasoningLinks.push_back({cid, l, conf, ctx});
    if(reasoningLinks.size()>20) reasoningLinks.erase(reasoningLinks.begin());
}
void Neuron::pruneWeak(){
    vector<DynamicEdge> ni,no;
    for(auto& e:inputs) if(e.total()>hp.PRUNE_WEIGHT_THRESH) ni.push_back(e);
    for(auto& e:outputs) if(e.total()>hp.PRUNE_WEIGHT_THRESH) no.push_back(e);
    inputs.swap(ni); outputs.swap(no);
}
void Neuron::updateEdges(){ for(auto& e:inputs) e.decay(); for(auto& e:outputs) e.decay(); pruneWeak(); }
// 计算两个语境快照之间的相似度（基于字段匹配）
float computeContextSimilarity(const ContextSnapshot& a, const ContextSnapshot& b) {
    float score = 0.0f;
    if (a.prevPos == b.prevPos) score += 0.25f;
    if (a.nextPos == b.nextPos) score += 0.25f;
    if (a.logicType == b.logicType) score += 0.25f;
    if (a.isStart == b.isStart) score += 0.25f;
    // 忽略 prevToken 和 nextToken，因为它们是具体 token ID，变化太敏感
    return score;  // 范围 0.0 ~ 1.0
}
std::vector<std::pair<int, float>> Neuron::getEdgesWithMatchScore(int contextID) const {
    std::vector<std::pair<int, float>> result;
    result.reserve(outputs.size());

    // 先找出当前语境下哪些边被记录（即目标token在contextBuckets[contextID]中）
    std::unordered_set<int> activeEdgeIndices;
    if (contextID < (int)contextBuckets.size()) {
        const auto& targets = contextBuckets[contextID];
        for (int target : targets) {
            for (int i = 0; i < (int)outputs.size(); ++i) {
                if (outputs[i].target == target) {
                    activeEdgeIndices.insert(i);
                    break;  // 每个目标只匹配第一条边（通常唯一）
                }
            }
        }
    }

    // 为每条边分配匹配度
    for (int i = 0; i < (int)outputs.size(); ++i) {
        // 如果该边在当前语境下被记录，匹配度为1.0，否则为0.2（基础传播）
        float matchScore = (activeEdgeIndices.count(i) > 0) ? 1.0f : 0.2f;
        result.emplace_back(i, matchScore);
    }
    return result;
}
void Neuron::addTargetToContext(int contextID, int targetToken) {
    if (contextID < 0 || targetToken <= 3) return;
    if (contextID >= (int)contextBuckets.size()) {
        contextBuckets.resize(contextID + 1);
    }
    auto& bucket = contextBuckets[contextID];
    for (int t : bucket) {
        if (t == targetToken) return;
    }
    bucket.push_back(targetToken);
}
// ===================== TextTokenizer 实现 =====================
TextTokenizer::TextTokenizer(){
    vocabList={"<pad>","<sos>","<eos>","<unk>"};
    tokenPosCache.resize(4, POS_UNKNOWN);
    tokenIsPunctCache.resize(4, false);
    tokenIsNewCharCache.resize(4, false);
}
int TextTokenizer::getTokenId(const string& s){ for(int i=0;i<(int)vocabList.size();++i) if(vocabList[i]==s) return i; return -1; }
vector<string> TextTokenizer::splitGBK(const string& s){
    vector<string> res; int n=s.size(),i=0;
    while(i<n){
        unsigned char c=(unsigned char)s[i];
        if(c<0x81){ res.push_back(s.substr(i,1)); i++; continue; }
        if(i+1>=n) break;
        string ch=s.substr(i,2);
        string c1=ch.substr(0,1),c2=ch.substr(1,1);
        if(!symbolFilter.count(c1)) radicalPool.insert(c1);
        if(!symbolFilter.count(c2)) radicalPool.insert(c2);
        res.push_back(ch); i+=2;
    }
    return res;
}
string TextTokenizer::composeNewChar(){
    if(radicalPool.size()<2) return "";
    vector<string> parts(radicalPool.begin(),radicalPool.end());
    for(int t=0;t<5;t++){
        int l=rand()%parts.size(), r=rand()%parts.size();
        string nb=parts[l]+parts[r];
        unsigned char b1=(unsigned char)nb[0],b2=(unsigned char)nb[1];
        if(b1>=0x81 && b1<=0xFE && b2>=0x40 && b2<=0xFE && b2!=0x7F) return nb;
    }
    return "";
}
vector<int> TextTokenizer::encode(const string& text){
    lock_guard<recursive_mutex> lock(tokenMtx);
    vector<int> seq{1};
    auto chars=splitGBK(text);
    for(auto& ch:chars){
        int tid=getTokenId(ch);
        if(tid==-1){
            if((int)vocabList.size()>=hp.VOCAB_SIZE) continue;
            vocabList.push_back(ch);
            tid=vocabList.size()-1;
            if(tokenEmbedding.find(tid)==tokenEmbedding.end()) tokenEmbedding[tid]=vector<float>(EMBED_DIM,0.01f);
            tokenPosCache.push_back(POS_UNKNOWN);
            tokenIsPunctCache.push_back(symbolFilter.count(ch));
            tokenIsNewCharCache.push_back(isNewChar(ch));
        }
        seq.push_back(tid);
    }
    return seq;
}
string TextTokenizer::decode(const vector<int>& seq,int depth){
    if(depth>hp.MAX_DECODE_DEPTH) return "";
    lock_guard<recursive_mutex> lock(tokenMtx);
    string text;
    for(int tid:seq) if(tid>3 && tid<(int)vocabList.size()) text+=vocabList[tid];
    return text;
}
void TextTokenizer::buildVocab(const string& text){
    lock_guard<recursive_mutex> lock(tokenMtx);
    auto cs=splitGBK(text);
    for(auto& ch:cs){
        int tid=getTokenId(ch);
        if(tid==-1 && (int)vocabList.size()<hp.VOCAB_SIZE){
            vocabList.push_back(ch);
            tid=vocabList.size()-1;
            if(tokenEmbedding.find(tid)==tokenEmbedding.end()) tokenEmbedding[tid]=vector<float>(EMBED_DIM,0.01f);
            tokenPosCache.push_back(POS_UNKNOWN);
            tokenIsPunctCache.push_back(symbolFilter.count(ch));
            tokenIsNewCharCache.push_back(isNewChar(ch));
        }
    }
}
int TextTokenizer::createConcept(const vector<int>& tokens,const string& name){
    for(int i=0;i<(int)conceptSeq.size();++i) if(conceptSeq[i]==tokens) return i+nextConceptId;
    conceptSeq.push_back(tokens);
    conceptContent.emplace_back();
    return nextConceptId+(int)conceptSeq.size()-1;
}
bool TextTokenizer::isStablePhrase(const vector<int>& gram){
    if(gram.size()<2) return false;
    string first=vocabList[gram[0]];
    unordered_set<string> aux={"的","了","是","也","就","都","要","在","和","这","那"};
    if(aux.count(first)) return false;
    string last=vocabList[gram.back()];
    if(aux.count(last) && gram.size()>=2) return false;
    unordered_set<int> s(gram.begin(),gram.end());
    return s.size()==gram.size();
}
AutoPosType TextTokenizer::getPosForToken(int tid) const {
    // 优先从缓存读取
    auto it = posCache.find(tid);
    if (it != posCache.end()) {
        return it->second;
    }

    // 回退到旧的 tokenPosCache
    if (tid >= 0 && tid < (int)tokenPosCache.size()) {
        return tokenPosCache[tid];
    }

    return POS_UNKNOWN;
}
void TextTokenizer::setPosForToken(int tid, AutoPosType pos) {
    // 写入缓存
    if (tid >= 0) {
        posCache[tid] = pos;
    }

    // 同时写入旧的 tokenPosCache（兼容旧逻辑）
    if (tid >= 0 && tid < (int)tokenPosCache.size()) {
        tokenPosCache[tid] = pos;
    }
}
vector<int> TextTokenizer::extractConceptsFromSeq(const vector<int>& seq){
    vector<int> concepts;
    if(seq.empty()) return concepts;
    int n=seq.size();
    for(int i=0;i<n-1;i++){
        int a=seq[i], b=seq[i+1];
        if(a<=3 || b<=3) continue;
        AutoPosType pa=getPosForToken(a), pb=getPosForToken(b);
        if((pa==POS_ADJ && pb==POS_NOUN) ||
           (pa==POS_VERB && pb==POS_NOUN) ||
           (pa==POS_NOUN && pb==POS_NOUN)){
            vector<int> phrase={a,b};
            concepts.push_back(createConcept(phrase,"phrase"));
        }
    }
    for(int i=0;i<n;i++){
        if(i>=(int)vocabList.size()) break;
        if(isPunctuation(vocabList[seq[i]])) continue;
        for(int len=2;len<=3;len++){
            if(i+len>n) break;
            bool ok=true; vector<int> gram;
            for(int j=0;j<len;j++){ int t=seq[i+j]; if(t<=3 || isPunctuation(vocabList[t])){ ok=false; break; } gram.push_back(t); }
            if(ok && isStablePhrase(gram)) concepts.push_back(createConcept(gram,"auto_phrase"));
        }
    }
    return concepts;
}
vector<int> TextTokenizer::matchKnowledgeConcepts(const string& input){
    lock_guard<recursive_mutex> lock(tokenMtx);
    vector<int> res;
    for(auto& kv:knowledgeVec) if(input.find(kv.first)!=string::npos){ vector<int> tok=encode(kv.first); res.push_back(createConcept(tok,"knowledge")); }
    return res;
}
vector<int> TextTokenizer::getKnowledgeContent(int conceptId){
    lock_guard<recursive_mutex> lock(tokenMtx);
    int idx=conceptId-nextConceptId;
    if(idx<0 || idx>=(int)conceptSeq.size()) return {};
    return conceptContent[idx];
}
vector<int> TextTokenizer::splitCompoundConcepts(const string& input){
    lock_guard<recursive_mutex> lock(tokenMtx);
    vector<int> res;
    for(auto& kv:knowledgeVec) if(input.find(kv.first)!=string::npos){ vector<int> tok=encode(kv.first); res.push_back(createConcept(tok,"knowledge")); }
    return res;
}
bool TextTokenizer::isNewChar(const string& s){ if(numSet.count(s) || s.size()!=2) return false; return radicalPool.count(s.substr(0,1)) && radicalPool.count(s.substr(1,1)); }
bool TextTokenizer::isPunctuation(const string& s){ return symbolFilter.count(s); }
float TextTokenizer::cosineSimilarity(const vector<float>& a,const vector<float>& b){
    float dot=0,na=0,nb=0;
    for(int i=0;i<EMBED_DIM;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return dot/(sqrt(na)*sqrt(nb)+1e-8f);
}
void TextTokenizer::updateEmbeddings(){
    float lr=0.01f;
    for(auto& kv:transCount){
        int from=kv.first; auto& toMap=kv.second;
        if(tokenEmbedding.find(from)==tokenEmbedding.end()) tokenEmbedding[from]=vector<float>(EMBED_DIM,0.01f);
        for(auto& entry:toMap){
            int to=entry.first, cnt=entry.second;
            if(tokenEmbedding.find(to)==tokenEmbedding.end()) tokenEmbedding[to]=vector<float>(EMBED_DIM,0.01f);
            for(int i=0;i<EMBED_DIM;i++){
                float diff=tokenEmbedding[to][i]-tokenEmbedding[from][i];
                tokenEmbedding[from][i]+=lr*diff*cnt;
                tokenEmbedding[to][i]-=lr*diff*cnt;
            }
        }
    }
}
bool TextTokenizer::isHypothetical(int tid){
    if(tid<0 || tid>=(int)vocabList.size()) return false;
    static unordered_set<string> hypoWords={"如果","假如","若","假设","要是","倘若"};
    return hypoWords.count(vocabList[tid]);
}
bool TextTokenizer::isChronological(int from,int to){
    int fwd=transCount[from][to], rev=transCount[to][from];
    return fwd>rev;
}
void TextTokenizer::updateTokenCache() {
    // 原有逻辑：更新 tokenPosCache, tokenIsPunctCache, tokenIsNewCharCache
    tokenPosCache.resize(vocabList.size(), POS_UNKNOWN);
    tokenIsPunctCache.resize(vocabList.size(), false);
    tokenIsNewCharCache.resize(vocabList.size(), false);

    for (size_t i = 0; i < vocabList.size(); i++) {
        tokenIsPunctCache[i] = symbolFilter.count(vocabList[i]);
        tokenIsNewCharCache[i] = isNewChar(vocabList[i]);
    }

    // ========== 新增：同步 posCache 到 tokenPosCache ==========
    for (const auto& entry : posCache) {
        int tid = entry.first;
        if (tid >= 0 && tid < (int)tokenPosCache.size()) {
            tokenPosCache[tid] = entry.second;
        }
    }
}
// ========== 模板提取与匹配 ==========
void TextTokenizer::extractTemplatesFromCorpus(const vector<int>& seq) {
    if (seq.size() < 2) return;
    unordered_map<uint64_t, int> phraseCount;
    unordered_map<uint64_t, int> skeletonCount;
    for (int len = 2; len <= 4; ++len) {
        for (size_t i = 0; i + len <= seq.size(); ++i) {
            bool hasPunct = false;
            uint64_t code = 0;
            vector<AutoPosType> posSeq;
            for (int j = 0; j < len; ++j) {
                int tid = seq[i+j];
                if (tid <= 3 || isPunctuation(vocabList[tid])) { hasPunct = true; break; }
                code = (code << 16) | (uint64_t)tid;
                AutoPosType pos = (tid < (int)tokenPosCache.size()) ? tokenPosCache[tid] : POS_UNKNOWN;
                posSeq.push_back(pos);
            }
            if (!hasPunct && len >= 2) {
                phraseCount[code]++;
                uint64_t skCode = encodePosSeq(posSeq);
                skeletonCount[skCode]++;
            }
        }
    }
    for (auto& kv : phraseCount) {
        if (kv.second >= 2) {
            uint64_t code = kv.first;
            vector<int> tokens;
            for (int i = 0; i < 4; ++i) {
                int t = (code >> (48 - i*16)) & 0xFFFF;
                if (t == 0) break;
                tokens.push_back(t);
            }
            phraseTemplates.emplace_back(tokens, kv.second);
            if (!tokens.empty()) {
                phrasePrefix1[tokens[0]].push_back(phraseTemplates.size()-1);
                if (tokens.size() >= 2) {
                    uint64_t key = ((uint64_t)tokens[0] << 16) | tokens[1];
                    phrasePrefix2[key].push_back(phraseTemplates.size()-1);
                }
            }
        }
    }
    for (auto& kv : skeletonCount) {
        if (kv.second >= 2) {
            posSkeletonTemplates.emplace_back(kv.first, kv.second);
        }
    }
}

void TextTokenizer::extractTemplatesFromExcellentFile() {
    ifstream f(DYNAMIC_TRAIN_FILE);
    if (!f) return;
    string line;
    while (getline(f, line)) {
        if (line.empty() || line.size() < 3) continue;
        string gbkLine = utf8ToGbk(line);   // ← 转为 GBK
        vector<int> seq = encode(gbkLine);  // 现在 encode 得到正确 GBK 序列
        extractTemplatesFromCorpus(seq);
    }
    f.close();
}

vector<int> TextTokenizer::matchPhraseByPrefix(int token1, int token2) {
    vector<int> result;
    auto it1 = phrasePrefix1.find(token1);
    if (it1 == phrasePrefix1.end()) return result;
    for (int idx : it1->second) {
        auto& phrase = phraseTemplates[idx].first;
        if (phrase.size() >= 1 && phrase[0] == token1) {
            if (token2 == -1) {
                if (phrase.size() >= 2) result.push_back(phrase[1]);
            } else {
                if (phrase.size() >= 2 && phrase[1] == token2) {
                    if (phrase.size() >= 3) result.push_back(phrase[2]);
                }
            }
        }
    }
    return result;
}

AutoPosType TextTokenizer::matchPosSkeleton(const vector<AutoPosType>& recentPos) {
    if (recentPos.empty()) return POS_UNKNOWN;
    uint64_t key = encodePosSeq(recentPos);
    AutoPosType best = POS_UNKNOWN;
    int bestCount = 0;
    for (auto& sk : posSkeletonTemplates) {
        uint64_t skCode = sk.first;
        int matchLen = 0;
        uint64_t tmp = key, skTmp = skCode;
        while (tmp && skTmp) {
            uint8_t p1 = tmp & 0xFF;
            uint8_t p2 = skTmp & 0xFF;
            if (p1 == p2) matchLen++;
            else break;
            tmp >>= 8;
            skTmp >>= 8;
        }
        if (matchLen >= min(3, (int)recentPos.size()) && matchLen > bestCount) {
            bestCount = matchLen;
            uint8_t nextPos = (skCode >> (8*matchLen)) & 0xFF;
            best = (AutoPosType)nextPos;
        }
    }
    return best;
}
// ===================== BrainCortex 类定义 =====================
struct AttentionSubgraph {
    unordered_set<int> neurons;
    unordered_set<int> tokens;
    void clear() { neurons.clear(); tokens.clear(); }
};

class BrainCortex {
public:
	string getConsciousReport() { return generateConsciousReport(); }
    // ========== TD 学习相关 ==========
    vector<vector<DynamicEdge*>> stepEdges;
    vector<int> stepFluency;
    const int MAX_TRACE = 15;

    // 语境ID映射（用于训练和传播）
    unordered_map<uint64_t, int> contextHashToID;
    int getContextID(const ContextSnapshot& snap) {
        uint64_t h = snap.hash();
        auto it = contextHashToID.find(h);
        if (it != contextHashToID.end()) return it->second;
        int newID = (int)contextHashToID.size();
        contextHashToID[h] = newID;
        return newID;
    }

    void recordStepEdges(const vector<DynamicEdge*>& edges);
    void applyTDUpdates();
    void clearStepRecords();
	void recordTokenPosition(int tokenId, int contextID, int positionPercent);
    int calcPositionalScore(int tokenId, int contextID, int currentPosPct);
    std::string getPositionSummary(int tokenId, int contextID);
    vector<vector<Neuron>> layers;
    deque<int> shortMemory;
    deque<int> amygdalaActivation;
    unordered_set<int> longTermMemory;
    EmotionState curEmo;
    int goalIntent=-1;
    StyleIntent curStyle=STYLE_STORY;
    int nextNeuronId=1;
    unordered_set<int> knowledgeConcepts;
    deque<vector<int>> longTextMemory;
    unordered_map<int,LogicRelation> logicMap;
    vector<int> coreThemeTokens;
    deque<int> logicAnchorTokens;
    deque<Episode> episodicMemory;
    GlobalWorkspace workspace;
    float reasoningConfidenceThreshold = 0.3f;
    int generationCounter = 0;
    int neurogenesisCooldown = 0;
    float lastInferenceConfidence = 0.0f;

    struct SelfNeuron {
        string name;
        float value;
        float predicted;
        float error;
    };
    vector<SelfNeuron> selfNeurons;
    float avgPredictionError = 0.0f;
    deque<int> qualityHistoryForSelf;

    struct InnerGoal {
        string description;
        int targetToken;
        float progress;
        int assignedStep;
        bool active;
    };
    vector<InnerGoal> innerGoals;
    int stepCounter = 0;
    string lastConsciousReport;

    AttentionSubgraph curSubgraph;
    int attentionWindowSize = 3;
    int maxSubgraphNeurons = 500;
    int maxSubgraphTokens = 1000;

    vector<unordered_map<int, int>> layerIdToIndex;
    unordered_map<int, Neuron*> globalNeuronMap;

    WorldSimulator world;
    
    deque<PredictionRecord> predictionHistory;
    const int PREDICTION_HISTORY_SIZE = 100;
    float avgSurprise = 0.5f;

    // ========== 误差传播函数 ==========
    void propagateTokenError(int selectedToken, float surprise, 
                             const vector<int>& context, 
                             TextTokenizer& token);
    void triggerBackwardReasoning(int fromToken, int toToken, 
                                  const vector<int>& context,
                                  TextTokenizer& token);
    void reevaluateContext(const vector<int>& context, TextTokenizer& token);
    void integrateGlobalPredictionError();
    void updateEdgeWithError(DynamicEdge* edge, float error, int fromToken, int toToken, TextTokenizer& token);
    ContextSnapshot buildSnapshot(const vector<int>& context, int currentToken, TextTokenizer& token);

    // ========== 两环动态控制器 ==========
    int T=50, E=20, R=10;
    deque<int> T_history, E_history, R_history;
    int M1=30, M2=40;
    const int alpha1=15, beta1=3, gamma1=1;
    const int alpha2=12, beta2=2, gamma2=1;
    const int alpha3=10, beta3=1, gamma3=2;
    const int delta1=8, epsilon1=2, zeta1=1;
    const int delta2=9, epsilon2=1, zeta2=1;

    // ========== 模板匹配与类比生成 ==========
    int calcTemplateScore(const vector<int>& ctx, int candidate, TextTokenizer& token);
    vector<int> analogicalGenerate(const vector<int>& context, TextTokenizer& token);

    // ========== 四核架构新增成员 ==========
    float alpha_L = 1.0f;
    float alpha_A = 1.0f;
    float alpha_R = 1.0f;
    int deadlockCounter = 0;
    const int DEADLOCK_THRESHOLD = 3;

    std::unordered_map<int, int> calcLogicScore(const std::vector<int>& ctx, TextTokenizer& token);
    std::unordered_map<int, int> calcAssocScore(const std::vector<int>& ctx, TextTokenizer& token);
    std::unordered_map<int, int> calcRewardScore(const std::vector<int>& ctx, TextTokenizer& token);
    void updateGains(const std::vector<int>& ctx, TextTokenizer& token);

    // ========== 原有函数 ==========
    BrainCortex();
    void rebuildLayerIndex();
    int createNeuron(int layerIdx);
    Neuron* findNeuron(int layerIdx, int nid);
    Neuron* findNeuronByToken(int tid);
    Neuron* findNeuronById(int nid);
    int getOrBindNeuron(int tid, int layerIdx);
    void addToShortMemory(int tid);
    void saveLongTextSegment(const vector<int>& seg);
    void activateAmygdala(int tid);
    void updateGlobalMood();
    void emotionModulate();
    void layerForwardPass(TextTokenizer& token);
    void injectConcepts(const vector<int>& concepts);
    void injectKnowledge(int conceptId, const vector<int>& content, TextTokenizer& tok);
    bool isLogicalConsistent(int fromToken,int toToken,LogicRelation rel,const vector<int>& context,TextTokenizer& token);
    vector<int> runReasoning(const vector<int>& context,int maxDepth,TextTokenizer& token);
    void storeEpisode(const vector<int>& seq, TextTokenizer& token);
    vector<int> retrieveEpisodes(const vector<int>& context, TextTokenizer& token);
    vector<int> planToGoal(const vector<int>& start, int goalToken, int maxSteps, TextTokenizer& token);
    void autoInduceRelationsLocked(TextTokenizer& token);
    void autoInduceRelations(TextTokenizer& token);
    void learnLogicRelations(const vector<int>& seq, TextTokenizer& token);
    void autoLearnAllPos(const vector<int>& seq, TextTokenizer& token);
    void selfGrowth(const vector<int>& seq, TextTokenizer& token);
    void updateAllLayers();
    void pruneWeakNodes();
    int globalAttention(const vector<int>& ctx,int targetTid);
    int calcScore(const vector<int>& ctx, int targetTid, TextTokenizer& token);
    int calcFluency(const vector<int>& seq,TextTokenizer& token);
    bool checkConsistency(const vector<int>& seq, TextTokenizer& token);
    int evaluateOutput(const vector<int>& seq,TextTokenizer& token,int mode=3);
    bool deepReflect(const vector<int>& seq, int quality, TextTokenizer& token);
    bool reflect(const vector<int>& seq,TextTokenizer& token);
    void setGoal(int g);
    void setStyle(StyleIntent s);
    int totalNeurons();
    void addRandomNeuron();
    void maybeSplitLayer();
    void rewiring();
    void maybeMutateTopology(int lastQuality);
    void initSelfModel();
    void setSelfValue(const string& name, float val);
    float getSelfValue(const string& name) const;
    void updateSelfModel(int lastQuality, float inferenceConf, float topoChangeRate);
    void updateTwoLoopController(int quality);
    string generateConsciousReport();
    void generateInnerGoals();
    bool hasGoal(const string& desc);
    void consciousBroadcast();
    void buildAttentionSubgraph(const vector<int>& ctx, TextTokenizer& token);
    int calcScoreSubgraph(const vector<int>& ctx, int targetTid, TextTokenizer& token);
    void offlineConsolidation(TextTokenizer& token, int maxSeconds = 300);

    // ========== 新的四核 singleGenerate（声明） ==========
    vector<int> singleGenerate(vector<int> ctx, TextTokenizer& token);
    // ========== 新增：状态查询 ==========
    int getTotalNeurons() const { 
        int sum = 0;
        for (const auto& layer : layers) sum += layer.size();
        return sum;
    }
    string getEmotionStatus() const {
        const char* emoName[] = {"NORMAL", "EXCITE", "LOW", "QUEST"};
        return string(emoName[curEmo.type]) + " (" + to_string(curEmo.intensity) + ")";
    }
    string getStyleName() const {
        const char* styleName[] = {"STORY", "EMOTION", "SIMPLE"};
        return styleName[curStyle];
    }
    int getGoalIntent() const { return goalIntent; }
    
    // ========== 新增：在线学习 ==========
    void onlineLearn(const string& text, TextTokenizer& tokenizer) {
        auto seq = tokenizer.encode(text);
        selfGrowth(seq, tokenizer);
    }

    // ========== 激活边记录（安全版：存神经元ID+目标token） ==========
    vector<pair<int, int>> activatedEdgeKeys;  // {neuronId, targetToken}
    vector<int> activatedTokens;
    
    void clearActivatedEdges() {
        activatedEdgeKeys.clear();
        activatedTokens.clear();
    }
    
    void recordActivatedEdge(int neuronId, int targetToken) {
        if (neuronId >= 0 && targetToken > 3) {
            activatedEdgeKeys.emplace_back(neuronId, targetToken);
        }
    }
    
    void recordActivatedToken(int tid) {
        if (tid > 3) activatedTokens.push_back(tid);
    }
    
    // 通过 (neuronId, targetToken) 查找边
    DynamicEdge* findEdgeByKey(int neuronId, int targetToken) {
        Neuron* neu = findNeuronById(neuronId);
        if (!neu) return nullptr;
        return neu->findOutput(targetToken);
    }
    
    void reinforceActivatedEdges(int delta = 5) {
        if (activatedEdgeKeys.empty()) {
            cout << "[反馈] 没有激活的边可强化" << endl;
            return;
        }
        
        int count = 0;
        for (auto& key : activatedEdgeKeys) {
            DynamicEdge* edge = findEdgeByKey(key.first, key.second);
            if (!edge) continue;
            
            edge->weight += delta;
            edge->lifeCycle = hp.EDGE_DECAY_STEP;
            edge->decayAge = 0;
            count++;
            
            if (edge->weight >= hp.PERMANENT_WEIGHT_THRESH) {
                edge->permanent += 1;
                edge->weight = edge->weight % hp.PERMANENT_WEIGHT_THRESH;
                if (edge->weight < 5 && edge->permanent > 0) edge->weight += 5;
            }
        }
        
        cout << "[反馈] ? 强化了 " << count << " 条激活边 (+" << delta << ")" << endl;
        clearActivatedEdges();
    }
    
    void punishActivatedEdges(int delta = 5) {
        if (activatedEdgeKeys.empty()) {
            cout << "[反馈] 没有激活的边可惩罚" << endl;
            return;
        }
        
        int count = 0;
        for (auto& key : activatedEdgeKeys) {
            DynamicEdge* edge = findEdgeByKey(key.first, key.second);
            if (!edge) continue;
            edge->weight = max(0, edge->weight - delta);
            count++;
        }
        
        cout << "[反馈] ? 惩罚了 " << count << " 条激活边 (-" << delta << ")" << endl;
        clearActivatedEdges();
    }
};
// ========== 位置轨迹相关函数 ==========

// 记录一个token在某个语境下的位置百分比
void BrainCortex::recordTokenPosition(int tokenId, int contextID, int positionPercent) {
    if (tokenId <= 3 || contextID < 0) return;
    Neuron* neu = findNeuronByToken(tokenId);
    if (!neu) {
        int nid = getOrBindNeuron(tokenId, 0);
        neu = findNeuron(0, nid);
        if (!neu) return;
    }
    neu->recordPosition(contextID, positionPercent);
}

// 计算一个token在当前位置下的位置匹配得分
int BrainCortex::calcPositionalScore(int tokenId, int contextID, int currentPosPct) {
    if (tokenId <= 3 || contextID < 0) return 0;
    Neuron* neu = findNeuronByToken(tokenId);
    if (!neu) return 0;
    return neu->calcPositionMatchScore(contextID, currentPosPct);
}

// 获取位置摘要（调试用）
string BrainCortex::getPositionSummary(int tokenId, int contextID) {
    if (tokenId <= 3 || contextID < 0) return "（无效token）";
    Neuron* neu = findNeuronByToken(tokenId);
    if (!neu) return "（未找到神经元）";

    const vector<int>* positions = neu->getPositionsForContext(contextID);
    if (!positions || positions->empty()) return "（无位置数据）";

    int sum = 0, minPos = 100, maxPos = 0;
    for (int p : *positions) {
        sum += p;
        if (p < minPos) minPos = p;
        if (p > maxPos) maxPos = p;
    }
    float avg = (float)sum / positions->size();

    stringstream ss;
    ss << "出现" << positions->size() << "次，位置范围 " << minPos << "%~" << maxPos << "%，平均 " << (int)avg << "%";
    return ss.str();
}
// ===================== BrainCortex 实现 =====================
// 构建语境快照
ContextSnapshot BrainCortex::buildSnapshot(const vector<int>& context, int currentToken, TextTokenizer& token) {
    ContextSnapshot snap;
    if (!context.empty()) {
        snap.prevToken = (context.size() >= 2) ? context[context.size()-2] : -1;
        snap.nextToken = currentToken;
        snap.prevPos = (snap.prevToken >= 0 && snap.prevToken < (int)token.tokenPosCache.size()) ?
                       token.tokenPosCache[snap.prevToken] : POS_UNKNOWN;
        snap.nextPos = (currentToken >= 0 && currentToken < (int)token.tokenPosCache.size()) ?
                       token.tokenPosCache[currentToken] : POS_UNKNOWN;
        snap.logicType = LOGIC_NONE;
        snap.isStart = (context.size() == 1);
        snap.isEnd = false;
    } else {
        snap.prevPos = POS_UNKNOWN;
        snap.nextPos = POS_UNKNOWN;
        snap.logicType = LOGIC_NONE;
        snap.isStart = true;
        snap.isEnd = false;
    }
    return snap;
}

// 更新边带误差
void BrainCortex::updateEdgeWithError(DynamicEdge* edge, float error, int fromToken, int toToken, TextTokenizer& token) {
    if (!edge) return;
    
    // 记录误差
    edge->updateError(error);
    
    // 根据误差调整权重
    if (error > 2.0f) {
        // 高误差：强化（学习新规律）
        int boostAmount = (int)(error * 0.3f) + 1;
        edge->weight += boostAmount;
        
        // 记录语境
        vector<int> ctx = {fromToken, toToken};
        ContextSnapshot snap = buildSnapshot(ctx, toToken, token);
        edge->addContext(snap, (int)(error * 0.5f));
        
    } else if (error < 0.5f && edge->errorHistory.size() > 10) {
        // 低误差且稳定：略微弱化（避免过拟合）
        edge->weight = max(0, edge->weight - 1);
    }
    
    // 限制权重
    if (edge->weight > 200) edge->weight = 200;
}

// 主误差传播函数
void BrainCortex::propagateTokenError(int selectedToken, float surprise, 
                                      const vector<int>& context, 
                                      TextTokenizer& token) {
    if (context.empty() || selectedToken <= 3) return;
    
    int lastToken = context.back();
    Neuron* prevNeu = findNeuronByToken(lastToken);
    if (!prevNeu) return;
    
    // 1. 更新前一个神经元的预测置信度
    prevNeu->predictionConfidence = 1.0f - min(1.0f, surprise / 10.0f);
    
    // 2. 找到指向selectedToken的边
    DynamicEdge* edge = prevNeu->findOutput(selectedToken);
    if (!edge) return;
    
    // 3. 更新边误差
    updateEdgeWithError(edge, surprise, lastToken, selectedToken, token);
    
    // 4. 如果surprise高（意外），触发反向推理
    if (surprise > 3.0f) {
        triggerBackwardReasoning(lastToken, selectedToken, context, token);
    }
    
    // 5. 误差传播到更早的上下文（衰减）
    if (context.size() >= 2 && surprise > 1.0f) {
        int earlierToken = context[context.size() - 2];
        Neuron* earlierNeu = findNeuronByToken(earlierToken);
        if (earlierNeu) {
            float dilutedError = surprise * 0.3f / context.size();
            earlierNeu->errorAccumulator += dilutedError;
            
            // 如果累积误差过大，触发"重新评估"
            if (earlierNeu->errorAccumulator > 8.0f) {
                reevaluateContext(context, token);
                earlierNeu->errorAccumulator = 0;
            }
        }
    }
    
    // 6. 更新全局平均惊喜度
    avgSurprise = avgSurprise * 0.9f + surprise * 0.1f;
}

// 反向推理
void BrainCortex::triggerBackwardReasoning(int fromToken, int toToken, 
                                           const vector<int>& context,
                                           TextTokenizer& token) {
    // 获取源神经元
    Neuron* prevNeu = findNeuronByToken(fromToken);
    if (!prevNeu) return;
    
    // 1. 检查其他候选token
    vector<pair<int, int>> otherCandidates;
    for (auto& edge : prevNeu->outputs) {
        if (edge.target == toToken) continue;
        if (edge.total() > 5) {
            otherCandidates.push_back({edge.target, edge.total()});
        }
    }
    
    // 2. 如果存在更强的连接，建立语境期望
    if (!otherCandidates.empty()) {
        sort(otherCandidates.begin(), otherCandidates.end(),
             [](auto& a, auto& b) { return a.second > b.second; });
        
        int expectedToken = otherCandidates[0].first;
        ContextSnapshot snap = buildSnapshot(context, toToken, token);
        prevNeu->addContextExpectation(snap, expectedToken);
        
        // 记录到调试输出
        if (expectedToken >= 0 && expectedToken < (int)token.vocabList.size() &&
            toToken >= 0 && toToken < (int)token.vocabList.size()) {
            cout << "[反向推理] 语境中期望 '" 
                 << token.vocabList[expectedToken] 
                 << "' 但选择了 '" << token.vocabList[toToken] << "'" << endl;
        }
    }
    
    // 3. 如果存在前向连接，传播误差
    if (context.size() >= 2) {
        int earlierToken = context[context.size() - 2];
        Neuron* earlierNeu = findNeuronByToken(earlierToken);
        if (earlierNeu) {
            float error = 2.0f / context.size();
            earlierNeu->errorAccumulator += error;
        }
    }
}

// 上下文重新评估
void BrainCortex::reevaluateContext(const vector<int>& context, TextTokenizer& token) {
    if (context.size() < 2) return;
    
    // 1. 对上下文中的每个边，检查合理性
    for (int i = 0; i < (int)context.size() - 1; i++) {
        int from = context[i];
        int to = context[i + 1];
        if (from <= 3 || to <= 3) continue;
        
        Neuron* neu = findNeuronByToken(from);
        if (!neu) continue;
        
        DynamicEdge* edge = neu->findOutput(to);
        if (!edge) continue;
        
        // 2. 计算该边在当前语境下的合理性
        ContextSnapshot snap = buildSnapshot(context, to, token);
        int contextWeight = edge->getContextWeight(snap);
        int baseWeight = edge->weight;
        
        float rationality = (float)contextWeight / (baseWeight + 1);
        
        // 3. 调整
        if (rationality < 0.2f && baseWeight > 10) {
            // 在其他语境下很强，当前语境很弱 → 适应当前语境
            edge->addContext(snap, 8);
            edge->weight += 1;
        }
        
        if (rationality > 3.0f && baseWeight < 30) {
            // 在当前语境下异常强 → 提升基础权重
            edge->weight += 2;
        }
    }
}

// 全局误差整合
void BrainCortex::integrateGlobalPredictionError() {
    // 1. 检查近期预测记录
    if (predictionHistory.empty()) return;
    
    float recentSurprise = 0;
    int count = min(20, (int)predictionHistory.size());
    auto it = predictionHistory.end();
    for (int i = 0; i < count; i++) {
        --it;
        recentSurprise += it->surprise;
    }
    recentSurprise /= count;
    
    // 2. 根据误差调整拓扑
    if (recentSurprise > 4.0f && neurogenesisCooldown <= 0) {
        // 高惊喜 → 增加神经元（探索新模式）
        addRandomNeuron();
        neurogenesisCooldown = hp.NEUROGENESIS_COOLDOWN;
        cout << "[全局误差] 高惊喜度(" << recentSurprise 
             << ")，新增神经元探索新模式" << endl;
    }
    
    if (recentSurprise < 0.3f) {
        // 低惊喜 → 修剪冗余连接
        int pruned = 0;
        for (auto& layer : layers) {
            for (auto& neu : layer) {
                for (auto it = neu.outputs.begin(); it != neu.outputs.end();) {
                    if (it->weight < 2 && it->permanent == 0 && it->predictionError < 1.0f) {
                        it = neu.outputs.erase(it);
                        pruned++;
                    } else {
                        ++it;
                    }
                }
            }
        }
        if (pruned > 0) {
            cout << "[全局误差] 低惊喜度(" << recentSurprise 
                 << ")，修剪了 " << pruned << " 条冗余边" << endl;
        }
    }
    
    // 3. 更新自我模型
    setSelfValue("PredictionErrorGlobal", recentSurprise);
    setSelfValue("SurpriseAvg", recentSurprise);
    
    // 4. 如果预测误差持续偏高，调整温度
    if (recentSurprise > 5.0f && predictionHistory.size() > 10) {
        // 提高探索性
        hp.GEN_TEMP = min(90, hp.GEN_TEMP + 2);
        cout << "[全局误差] 持续高误差，提高生成温度到 " << hp.GEN_TEMP << endl;
    }
}
void BrainCortex::recordStepEdges(const vector<DynamicEdge*>& edges) {
    stepEdges.push_back(edges);
}

void BrainCortex::clearStepRecords() {
    stepEdges.clear();
    stepFluency.clear();
}

void BrainCortex::applyTDUpdates() {
    if (stepEdges.size() < 2) return;
    int numSteps = stepEdges.size();
    for (int t = 1; t < numSteps; ++t) {
        int delta = stepFluency[t] - stepFluency[t-1];
        if (delta == 0) continue;
        for (int s = 0; s < t; ++s) {
            int age = t - s;
            if (age > MAX_TRACE) continue;
            int adjust = delta * (MAX_TRACE - age) / 10;
            if (adjust == 0) continue;
            for (auto* edge : stepEdges[s]) {
                if (!edge) continue;
                edge->weight += adjust;
                if (edge->weight < 0) edge->weight = 0;
                if (edge->weight >= hp.PERMANENT_WEIGHT_THRESH) {
                    edge->permanent += edge->weight / 2;
                    edge->weight /= 2;
                }
                edge->lifeCycle = hp.EDGE_DECAY_STEP;
                edge->decayAge = 0; 
            }
        }
    }
    clearStepRecords();
}

BrainCortex::BrainCortex() {
    layers.resize(hp.CORTEX_LAYERS);
    layerIdToIndex.resize(hp.CORTEX_LAYERS);
    initSelfModel();
    stepCounter = 0;
    lastConsciousReport = "";
}

void BrainCortex::rebuildLayerIndex() {
    globalNeuronMap.clear();
    for (int l = 0; l < (int)layers.size(); ++l) {
        layerIdToIndex[l].clear();
        for (int i = 0; i < (int)layers[l].size(); ++i) {
            int nid = layers[l][i].neuronId;
            layerIdToIndex[l][nid] = i;
            globalNeuronMap[nid] = &layers[l][i];
        }
    }
}

int BrainCortex::createNeuron(int layerIdx){
    if(layerIdx<0 || layerIdx>=hp.CORTEX_LAYERS) return -1;
    int nid=nextNeuronId++;
    layers[layerIdx].emplace_back(nid,layerIdx);
    Neuron& neu=layers[layerIdx].back();
    if(layerIdx==0) neu.setMode(MODE_LANGUAGE);
    else if(layerIdx==1) neu.setMode(MODE_ATTENTION);
    else if(layerIdx==2) neu.setMode(MODE_CONCEPT);
    else if(layerIdx==3) neu.setMode(MODE_LOGIC);
    else if(layerIdx==4) neu.setMode(MODE_MEMORY);
    else if(layerIdx==5) neu.setMode(MODE_EMOTION);
    layerIdToIndex[layerIdx][nid] = layers[layerIdx].size() - 1;
    globalNeuronMap[nid] = &layers[layerIdx].back();
    return nid;
}

Neuron* BrainCortex::findNeuron(int layerIdx, int nid){
    if(layerIdx<0 || layerIdx>=hp.CORTEX_LAYERS) return nullptr;
    auto it = layerIdToIndex[layerIdx].find(nid);
    if (it != layerIdToIndex[layerIdx].end()) return &layers[layerIdx][it->second];
    return nullptr;
}

Neuron* BrainCortex::findNeuronByToken(int tid){
    if(tid<0) return nullptr;
    for(auto& layer:layers) for(auto& neu:layer) if(neu.boundTokens.count(tid)) return &neu;
    return nullptr;
}

Neuron* BrainCortex::findNeuronById(int nid){
    auto it = globalNeuronMap.find(nid);
    if (it != globalNeuronMap.end()) return it->second;
    return nullptr;
}

int BrainCortex::getOrBindNeuron(int tid, int layerIdx){
    if(tid<0 || layerIdx<0 || layerIdx>=hp.CORTEX_LAYERS) return -1;
    for(auto& neu:layers[layerIdx]) if(neu.boundTokens.count(tid)) return neu.neuronId;
    int nid=createNeuron(layerIdx);
    Neuron* neu=findNeuron(layerIdx,nid);
    if(neu) neu->bindToken(tid);
    return nid;
}

void BrainCortex::addToShortMemory(int tid){ if(tid<0) return; shortMemory.push_back(tid); if(shortMemory.size()>hp.SHORT_MEMORY_SIZE) shortMemory.pop_front(); }

void BrainCortex::saveLongTextSegment(const vector<int>& seg){
    if(seg.empty()) return;
    longTextMemory.push_back(seg); if(longTextMemory.size()>10) longTextMemory.pop_front();
    for(int t:seg) { longTermMemory.insert(t); if(longTermMemory.size()>hp.LONG_TERM_LOGIC_MEMORY) longTermMemory.erase(longTermMemory.begin()); }
}

void BrainCortex::activateAmygdala(int tid){
    if(tid<0) return;
    lock_guard<recursive_mutex> lock(cortexMtx);
    for(auto& layer:layers) for(auto& neu:layer) if(neu.boundTokens.count(tid)){
        int pulse=neu.emotionEnergy+neu.activation;
        for(auto& e:neu.outputs) pulse+=e.total()/2;
        amygdalaActivation.push_back(pulse);
        if(amygdalaActivation.size()>hp.AMYGDALA_MEM_SIZE) amygdalaActivation.pop_front();
        updateGlobalMood();
        return;
    }
}

void BrainCortex::updateGlobalMood(){
    int sum=0,cnt=amygdalaActivation.size();
    if(cnt==0){ curEmo.intensity=0; curEmo.type=EMO_NORMAL; return; }
    for(int v:amygdalaActivation) sum+=v;
    curEmo.intensity=sum/cnt;
    if(curEmo.intensity>60) curEmo.type=EMO_EXCITE;
    else if(curEmo.intensity<5) curEmo.type=EMO_LOW;
    else curEmo.type=EMO_NORMAL;
    emotionModulate();
}

void BrainCortex::emotionModulate() {
    float factor = 1.0f;
    if (curEmo.type == EMO_EXCITE) factor = 1.5f;
    else if (curEmo.type == EMO_LOW) factor = 0.6f;
    reasoningConfidenceThreshold = 0.3f / factor;
    reasoningConfidenceThreshold = max(0.1f, min(0.8f, reasoningConfidenceThreshold));
}

void BrainCortex::layerForwardPass(TextTokenizer& token) {
    workspace.clear();
	logger.logDebug("layerForwardPass 进入");
    
    workspace.clear();
    logger.logDebug("workspace cleared");

    ContextSnapshot currentSnap;
    if (!shortMemory.empty()) {
        int lastToken = shortMemory.back();
        int prevToken = (shortMemory.size() >= 2) ? shortMemory[shortMemory.size()-2] : -1;
        currentSnap.prevToken = prevToken;
        currentSnap.nextToken = -1;
        currentSnap.prevPos = (prevToken >= 0 && prevToken < (int)token.tokenPosCache.size()) ?
                              token.tokenPosCache[prevToken] : POS_UNKNOWN;
        currentSnap.nextPos = POS_UNKNOWN;
        currentSnap.logicType = LOGIC_NONE;
        currentSnap.isStart = (shortMemory.size() == 1);
        currentSnap.isEnd = false;
    } else {
        currentSnap.prevPos = POS_UNKNOWN;
        currentSnap.nextPos = POS_UNKNOWN;
        currentSnap.logicType = LOGIC_NONE;
        currentSnap.isStart = true;
        currentSnap.isEnd = false;
    }
	logger.logDebug("[DEBUG] 语境快照构建完成");
    int currentContextID = getContextID(currentSnap);
    logger.logDebug("currentContextID = " + to_string(currentContextID));

    for (int lay = 0; lay < hp.CORTEX_LAYERS - 1; ++lay) {
        logger.logDebug("处理层 " + to_string(lay) + ", 神经元数=" + to_string(layers[lay].size()));
        
        const auto& currLayer = layers[lay];
        auto& nextLayer = layers[lay+1];
        const auto& nextIdxMap = layerIdToIndex[lay+1];

        for (auto& neu : nextLayer) neu.layerFeature = 0;
        logger.logDebug("层 " + to_string(lay) + " 初始化完成");

        for (int i = 0; i < (int)currLayer.size(); ++i) {
            const Neuron& neu = currLayer[i];
            if (neu.activation <= 0) continue;

            if (neu.activation > 30 && neu.getBestToken() != -1) {
                workspace.broadcast(neu.getBestToken(), neu.activation);
            }

            auto edgesWithScore = neu.getEdgesWithMatchScore(currentContextID);
            for (const auto& p : edgesWithScore) {
                int edgeIdx = p.first;
                float matchScore = p.second;
                if (edgeIdx < 0 || edgeIdx >= (int)neu.outputs.size()) continue;
                const DynamicEdge& edge = neu.outputs[edgeIdx];
                int tarNid = edge.target;
                auto it = nextIdxMap.find(tarNid);
                if (it == nextIdxMap.end()) continue;

                int contrib = (int)(edge.weight * neu.activation * matchScore / 8.0f);
                if (contrib < 0) continue;

                nextLayer[it->second].layerFeature += contrib;
            }
        }
        logger.logDebug("层 " + to_string(lay) + " 处理完成");
    }
    logger.logDebug("layerForwardPass 完成");
}


void BrainCortex::injectConcepts(const vector<int>& concepts){
    for(int cid:concepts){ getOrBindNeuron(cid,2); getOrBindNeuron(cid,3); }
    for(int i=0;i<(int)concepts.size();i++){
        int a=concepts[i], na=getOrBindNeuron(a,2);
        for(int j=max(0,i-2);j<i;j++){
            int b=concepts[j], nb=getOrBindNeuron(b,2);
            Neuron* nA=findNeuron(2,na); Neuron* nB=findNeuron(2,nb);
            if(nA) nA->linkOut(nb); if(nB) nB->linkOut(na);
        }
    }
    rebuildLayerIndex();
}

void BrainCortex::injectKnowledge(int conceptId, const vector<int>& content, TextTokenizer& tok){
    if(conceptId<0 || content.empty()) return;
    knowledgeConcepts.insert(conceptId);
    int nc=getOrBindNeuron(conceptId,3);
    Neuron* nC=findNeuron(3,nc); if(!nC) return;
    for(int tid:content){
        int nt=getOrBindNeuron(tid,3);
        Neuron* nT=findNeuron(3,nt);
        nC->linkOut(nt); if(nT) nT->linkIn(nc);
    }
    for(int i=0;i<(int)content.size()-1;i++){
        int a=content[i],b=content[i+1];
        int na=getOrBindNeuron(a,3), nb=getOrBindNeuron(b,3);
        Neuron* nA=findNeuron(3,na); if(nA) nA->linkOut(nb);
    }
    rebuildLayerIndex();
}

bool BrainCortex::isLogicalConsistent(int fromToken,int toToken,LogicRelation rel,const vector<int>& context,TextTokenizer& token){
    static unordered_map<string,string> antonym={{"好","坏"},{"大","小"},{"热","冷"},{"高","低"},{"多","少"}};
    string fromWord=(fromToken>=0 && fromToken<(int)token.vocabList.size())?token.vocabList[fromToken]:"";
    string toWord=(toToken>=0 && toToken<(int)token.vocabList.size())?token.vocabList[toToken]:"";
    if(rel==LOGIC_CAUSE && antonym.count(fromWord) && antonym[fromWord]==toWord) return false;
    if(rel==LOGIC_CAUSE && token.isChronological(toToken, fromToken)) return false;
    return true;
}

vector<int> BrainCortex::runReasoning(const vector<int>& context, int maxDepth, TextTokenizer& token) {
    if (context.empty() || maxDepth <= 0) return {};

    vector<float> ctxEmb(token.EMBED_DIM, 0.0f);
    int embCnt = 0;
    for (int t : context) {
        if (token.tokenEmbedding.find(t) != token.tokenEmbedding.end()) {
            for (int i = 0; i < token.EMBED_DIM; ++i) ctxEmb[i] += token.tokenEmbedding[t][i];
            embCnt++;
        }
    }
    if (embCnt > 0) {
        for (int i = 0; i < token.EMBED_DIM; ++i) ctxEmb[i] /= embCnt;
    }

    struct AStarState {
        int token;
        float g;
        float h;
        vector<int> path;
        int depth;
        vector<int> supporting;
        float f() const { return g + h; }
        bool operator<(const AStarState& other) const {
            return f() > other.f();
        }
    };

    priority_queue<AStarState> pq;
    unordered_set<int> visited;

    for (int startToken : context) {
        if (startToken <= 3) continue;
        float hVal = 0.0f;
        if (token.tokenEmbedding.find(startToken) != token.tokenEmbedding.end()) {
            hVal = 1.0f - token.cosineSimilarity(ctxEmb, token.tokenEmbedding[startToken]);
        }
        pq.push({startToken, 0.0f, hVal, {startToken}, 0, {startToken}});
        visited.insert(startToken);
    }

    vector<int> bestPath;
    float bestScore = -1e9;
    int iterations = 0;

    while (!pq.empty() && iterations < 500) {
        AStarState cur = pq.top(); pq.pop();
        iterations++;

        if (cur.depth >= maxDepth || cur.token == 2) {
            if (cur.g > bestScore) {
                bestScore = cur.g;
                bestPath = cur.path;
            }
            continue;
        }

        Neuron* neu = findNeuronByToken(cur.token);
        if (!neu) continue;

        for (const auto& edge : neu->outputs) {
            int nextToken = edge.target;
            if (nextToken <= 3) continue;
            float edgeCost = -log((float)edge.total() + 1.0f);
            float newG = cur.g + edgeCost;

            if (!isLogicalConsistent(cur.token, nextToken, edge.logic, context, token)) {
                continue;
            }

            if (visited.find(nextToken) != visited.end()) continue;
            visited.insert(nextToken);

            float hVal = 10.0f;
            if (token.tokenEmbedding.find(nextToken) != token.tokenEmbedding.end()) {
                float sim = token.cosineSimilarity(ctxEmb, token.tokenEmbedding[nextToken]);
                hVal = (1.0f - sim) * 5.0f;
                if (edge.logic != LOGIC_NONE) hVal *= 0.5f;
            }

            vector<int> newPath = cur.path;
            newPath.push_back(nextToken);
            vector<int> newSupport = cur.supporting;
            newSupport.push_back(nextToken);

            pq.push({nextToken, newG, hVal, newPath, cur.depth + 1, newSupport});
        }
    }

    if (!bestPath.empty()) {
        for (size_t i = 0; i < bestPath.size(); ++i) {
            Neuron* neu = findNeuronByToken(bestPath[i]);
            if (neu) {
                LogicRelation rel = LOGIC_NONE;
                if (i > 0) {
                    Neuron* prevNeu = findNeuronByToken(bestPath[i-1]);
                    if (prevNeu) {
                        auto* edge = prevNeu->findOutput(bestPath[i]);
                        if (edge) rel = edge->logic;
                    }
                }
                neu->addReasoning(bestPath[i], rel, 0.8f, bestPath);
            }
        }
        lastInferenceConfidence = min(1.0f, 1.0f / (1.0f + bestScore));
        if (bestPath.size() > context.size()) {
            vector<int> result(bestPath.begin() + context.size(), bestPath.end());
            return result;
        }
        return bestPath;
    }

    lastInferenceConfidence = 0.1f;
    return {};
}

void BrainCortex::storeEpisode(const vector<int>& seq, TextTokenizer& token) {
    if(seq.empty()) return;
    vector<float> emb(token.EMBED_DIM, 0.0f);
    int cnt = 0;
    for(int t : seq){
        if(token.tokenEmbedding.count(t)){
            auto& e = token.tokenEmbedding[t];
            for(int i=0; i<token.EMBED_DIM; i++) emb[i] += e[i];
            cnt++;
        }
    }
    if(cnt>0){
        for(int i=0; i<token.EMBED_DIM; i++) emb[i] /= cnt;
    }
    float importance = (float)seq.size() / 100.0f;
    episodicMemory.push_front({seq, emb, chrono::system_clock::now(), importance});
    if(episodicMemory.size() > hp.EPISODIC_MEMORY_LIMIT) episodicMemory.pop_back();
}

vector<int> BrainCortex::retrieveEpisodes(const vector<int>& context, TextTokenizer& token) {
    if(episodicMemory.empty() || context.empty()) return {};
    vector<float> ctxEmb(token.EMBED_DIM, 0.0f);
    int cnt = 0;
    for(int t : context){
        if(token.tokenEmbedding.count(t)){
            auto& e = token.tokenEmbedding[t];
            for(int i=0; i<token.EMBED_DIM; i++) ctxEmb[i] += e[i];
            cnt++;
        }
    }
    if(cnt==0) return {};
    for(int i=0; i<token.EMBED_DIM; i++) ctxEmb[i] /= cnt;
    float bestSim = -1.0;
    vector<int> bestTokens;
    for(auto& ep : episodicMemory){
        float sim = token.cosineSimilarity(ctxEmb, ep.embedding);
        if(sim > bestSim && sim > 0.5f){
            bestSim = sim;
            bestTokens = ep.tokens;
        }
    }
    return bestTokens;
}

vector<int> BrainCortex::planToGoal(const vector<int>& start, int goalToken, int maxSteps, TextTokenizer& token) {
    struct State {
        vector<int> seq;
        int token;
        int steps;
        float heuristic;
        bool operator<(const State& o) const { return heuristic > o.heuristic; }
    };
    priority_queue<State> pq;
    pq.push({start, start.back(), 0, 0.0f});
    unordered_set<int> visited;
    while(!pq.empty()){
        State cur = pq.top(); pq.pop();
        if(cur.token == goalToken) return cur.seq;
        if(cur.steps >= maxSteps) continue;
        Neuron* neu = findNeuronByToken(cur.token);
        if(!neu) continue;
        for(auto& edge : neu->outputs){
            int next = edge.target;
            if(visited.count(next)) continue;
            visited.insert(next);
            float sim = token.cosineSimilarity(token.tokenEmbedding[next], token.tokenEmbedding[goalToken]);
            float h = 1.0f - sim;
            vector<int> newSeq = cur.seq;
            newSeq.push_back(next);
            pq.push({newSeq, next, cur.steps+1, h + cur.steps*0.1f});
        }
    }
    return {};
}

void BrainCortex::autoInduceRelationsLocked(TextTokenizer& token) {
    token.updateEmbeddings();
    for (auto& kv : token.transCount) {
        int from = kv.first;
        auto& toMap = kv.second;
        for (auto& entry : toMap) {
            int to = entry.first;
            int cnt = entry.second;
            if (cnt < 3) continue;
            float pmi = log2((float)cnt * token.totalTransCount / (token.tokenTotalCount[from] * token.tokenTotalCount[to]));
            if (pmi < 1.5f) continue;
            vector<float>& embFrom = token.tokenEmbedding[from];
            vector<float>& embTo   = token.tokenEmbedding[to];
            float cosSim = token.cosineSimilarity(embFrom, embTo);
            LogicRelation inferred = LOGIC_NONE;
            if (pmi > 2.0f && cosSim > 0.5f && cosSim < 0.9f && token.isChronological(from, to))
                inferred = LOGIC_CAUSE;
            else if (cosSim < 0.2f && pmi > 1.0f)
                inferred = LOGIC_TURN;
            else if (token.isHypothetical(from) && pmi > 1.8f)
                inferred = LOGIC_CONDITION;
            if (inferred != LOGIC_NONE) {
                Neuron* fromNeu = findNeuronByToken(from);
                if (fromNeu) fromNeu->linkOut(to, inferred);
            }
        }
    }
    rebuildLayerIndex();
}

void BrainCortex::autoInduceRelations(TextTokenizer& token) {
    lock_guard<recursive_mutex> lock(tokenMtx);
    lock_guard<recursive_mutex> lockCortex(cortexMtx);
    autoInduceRelationsLocked(token);
}

void BrainCortex::learnLogicRelations(const vector<int>& seq, TextTokenizer& token) {
    lock_guard<recursive_mutex> lock(tokenMtx);
    lock_guard<recursive_mutex> lockCortex(cortexMtx);

    int n = (int)seq.size();
    if (n < 2) return;

    const int MAX_DIST = 1;                     // 只建相邻边
    const int SATURATION_LIMIT = 30;

    // 获取所有神经元的指针（用于快速访问）
    vector<Neuron*> neurons(n, nullptr);
    for (int i = 0; i < n; ++i) {
        int tid = seq[i];
        if (tid <= 3) continue;
        neurons[i] = findNeuronByToken(tid);
        if (!neurons[i]) {
            int nid = getOrBindNeuron(tid, 0);
            neurons[i] = findNeuron(0, nid);
        }
    }

    float maxPos = max(1, n - 1);  // 用于归一化位置

    for (int i = 0; i < n; ++i) {
        if (!neurons[i]) continue;
        Neuron* neuA = neurons[i];

        // ========== 新增：记录当前token的位置 ==========
        int posPct = (int)((float)i / maxPos * 100);
        // 构建当前语境的快照
        ContextSnapshot currentSnap;
        currentSnap.prevToken = (i >= 1) ? seq[i-1] : -1;
        currentSnap.nextToken = (i+1 < n) ? seq[i+1] : -1;
        currentSnap.prevPos = (currentSnap.prevToken >= 0 && currentSnap.prevToken < (int)token.tokenPosCache.size()) ?
                              token.tokenPosCache[currentSnap.prevToken] : POS_UNKNOWN;
        currentSnap.nextPos = (currentSnap.nextToken >= 0 && currentSnap.nextToken < (int)token.tokenPosCache.size()) ?
                              token.tokenPosCache[currentSnap.nextToken] : POS_UNKNOWN;
        currentSnap.logicType = LOGIC_NONE;
        currentSnap.isStart = (i == 0);
        currentSnap.isEnd = (i == n-1);
        int currentContextID = getContextID(currentSnap);
        recordTokenPosition(seq[i], currentContextID, posPct);

        // 原有逻辑：建立相邻边
        for (int j = i + 1; j < n && (j - i) <= MAX_DIST; ++j) {
            if (!neurons[j]) continue;
            int b = seq[j];

            // 语境快照（用于边）
            ContextSnapshot snap;
            snap.prevToken = (i >= 1) ? seq[i-1] : -1;
            snap.nextToken = (j+1 < n) ? seq[j+1] : -1;
            snap.prevPos = (snap.prevToken >= 0 && snap.prevToken < (int)token.tokenPosCache.size()) ?
                           token.tokenPosCache[snap.prevToken] : POS_UNKNOWN;
            snap.nextPos = (snap.nextToken >= 0 && snap.nextToken < (int)token.tokenPosCache.size()) ?
                           token.tokenPosCache[snap.nextToken] : POS_UNKNOWN;
            snap.logicType = LOGIC_NONE;
            snap.isStart = (i == 0);
            snap.isEnd = (j == n-1);
            int contextID = getContextID(snap);

            // ---------- 在 outputs 中建立边 ----------
            DynamicEdge* edge = neuA->findOutput(b);
            if (!edge) {
                neuA->outputs.emplace_back(b);
                edge = &neuA->outputs.back();
                edge->logic = LOGIC_NONE;
            }
            edge->weight = min(SATURATION_LIMIT, edge->weight + 1);

            // ---------- 将目标token加入神经元的语境桶 ----------
            neuA->addTargetToContext(contextID, b);
        }
    }

    // 自动归纳逻辑关系（保留原逻辑）
    static int lastTotal = 0;
    if (token.totalTransCount - lastTotal > 2000) {
        autoInduceRelationsLocked(token);
        lastTotal = token.totalTransCount;
    }
}

void BrainCortex::autoLearnAllPos(const vector<int>& seq, TextTokenizer& token) {
    // ========== 新版：直接使用外部词性标注 ==========
    // 不再使用规则和统计，直接从缓存读取

    for (int i = 0; i < (int)seq.size(); i++) {
        int tid = seq[i];
        if (tid <= 3) continue;  // 跳过 <pad>, <sos>, <eos>, <unk>

        AutoPosType pos = token.getPosForToken(tid);

        // 写入神经元
        Neuron* targetNode = findNeuronByToken(tid);
        if (targetNode) {
            targetNode->autoPos = pos;
        }

        token.setPosForToken(tid, pos);
    }

    token.updateTokenCache();
}

void BrainCortex::selfGrowth(const vector<int>& seq, TextTokenizer& token) {
    cout << "  提取概念..."; fflush(stdout);
    vector<int> concepts = token.extractConceptsFromSeq(seq);
    injectConcepts(concepts);
    cout << "完成\n  建立局部连接..."; fflush(stdout);

    int n = seq.size();
    for (int i = 0; i < n; i++) {
        int curTid = seq[i];
        if (curTid <= 3) continue;
        int curNeu = getOrBindNeuron(curTid, 0);
        for (int d = 1; d <= hp.MAX_LINK_OFFSET && i + d < n; d++) {
            int nextTid = seq[i + d];
            if (nextTid <= 3) continue;
            int nextNeu = getOrBindNeuron(nextTid, 0);
            Neuron* cNeu = findNeuron(0, curNeu);
            if (cNeu) cNeu->linkOut(nextNeu);
        }
        for (int d = hp.MAX_LINK_OFFSET + 1; d <= hp.LONG_RANGE_LINK && i + d < n; d++) {
            int farTid = seq[i + d];
            if (farTid <= 3) continue;
            int farNeu = getOrBindNeuron(farTid, 1);
            int curNeuId = getOrBindNeuron(curTid, 0);
            Neuron* cNeu = findNeuron(0, curNeuId);
            if (cNeu) cNeu->linkOut(farNeu);
        }
    }

    cout << "完成\n  学习逻辑关系..."; fflush(stdout);
    learnLogicRelations(seq, token);
    cout << "完成\n  更新所有层..."; fflush(stdout);
    updateAllLayers();
    pruneWeakNodes();
    cout << "完成\n  自动学习词性..."; fflush(stdout);

    // ========== 新增：先用外部词性分析器分析整段文本 ==========
    string fullText = token.decode(seq);
    token.analyzeText(fullText);

    // 再自动学习词性（从缓存读取）
    autoLearnAllPos(seq, token);

    token.updateTokenCache();
    cout << "完成" << endl;
    rebuildLayerIndex();
}

void BrainCortex::updateAllLayers(){ lock_guard<recursive_mutex> lock(cortexMtx); for(auto& layer:layers) for(auto& p:layer) p.updateEdges(); }

void BrainCortex::pruneWeakNodes(){
    lock_guard<recursive_mutex> lock(cortexMtx);
    for(auto& layer:layers){
        vector<int> rm;
        for(auto& p:layer) {
            if(p.inputs.empty() && p.outputs.empty() && p.maturity<5) rm.push_back(p.neuronId);
            else if(p.inactiveSteps > hp.INACTIVE_PRUNE_STEPS && p.boundTokens.empty() && p.maturity<10) rm.push_back(p.neuronId);
        }
        vector<Neuron> newLayer;
        for(auto& p:layer){
            bool del=false;
            for(int id:rm) if(p.neuronId==id){ del=true; break; }
            if(!del) newLayer.push_back(p);
        }
        layer.swap(newLayer);
    }
    rebuildLayerIndex();
}

int BrainCortex::globalAttention(const vector<int>& ctx,int targetTid){
    int score=0; if(ctx.empty() || targetTid<0) return 0;
    int start=max(0,(int)ctx.size()-hp.ATTENTION_SPAN);
    for(int i=start;i<(int)ctx.size();i++){
        int t=ctx[i], pos=i-start+1;
        for(auto& layer:layers) for(auto& neu:layer) if(neu.boundTokens.count(t)) for(auto& e:neu.outputs){
            int tarNid=e.target; for(auto& tNeu:layer) if(tNeu.neuronId==tarNid && tNeu.boundTokens.count(targetTid)){ score+=e.total()*pos; if(e.logic!=LOGIC_NONE) score+=3; }
        }
    }
    return score;
}

int BrainCortex::calcScore(const vector<int>& ctx, int targetTid, TextTokenizer& token) {
    lock_guard<recursive_mutex> lock(cortexMtx);
    if (targetTid < 0) return 0;

    // ========== 修改：如果子图已构建，使用子图评分 ==========
    if (!curSubgraph.tokens.empty() && !curSubgraph.tokens.count(targetTid)) {
        return 0;  // 不在子图中的token得分为0
    }

    int score = globalAttention(ctx, targetTid);

    for (auto& seg : longTextMemory) {
        for (int t : seg) {
            if (t == targetTid) score += 2;
        }
    }

    for (int t : shortMemory) {
        for (auto& layer : layers) {
            for (auto& p : layer) {
                if (p.boundTokens.count(t)) {
                    for (auto& e : p.outputs) {
                        int tarNid = e.target;
                        for (auto& tNeu : layer) {
                            if (tNeu.neuronId == tarNid && tNeu.boundTokens.count(targetTid)) {
                                score += e.total();
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto& layer : layers) {
        for (auto& p : layer) {
            if (p.boundTokens.count(targetTid)) {
                score += (p.emotionEnergy + curEmo.intensity) / 3;
                score += p.maturity / 4;
            }
        }
    }

    if (knowledgeConcepts.count(targetTid)) score += 15;
    if (targetTid == goalIntent) score *= 2;

    if (curStyle == STYLE_EMOTION) score = score * 12 / 10;
    if (curStyle == STYLE_SIMPLE) score = score * 8 / 10;

    if (!logicAnchorTokens.empty() && targetTid == logicAnchorTokens.back()) score += 30;

    vector<int> topWorkspace = workspace.getTop(3);
    if (find(topWorkspace.begin(), topWorkspace.end(), targetTid) != topWorkspace.end()) {
        score += 20;
    }

    // ========== 新增：位置轨迹打分 ==========
    if (!ctx.empty()) {
        ContextSnapshot currentSnap;
        int lastToken = ctx.back();
        int prevToken = (ctx.size() >= 2) ? ctx[ctx.size()-2] : -1;
        currentSnap.prevToken = prevToken;
        currentSnap.nextToken = -1;
        currentSnap.prevPos = (prevToken >= 0 && prevToken < (int)token.tokenPosCache.size()) ?
                              token.tokenPosCache[prevToken] : POS_UNKNOWN;
        currentSnap.nextPos = POS_UNKNOWN;
        currentSnap.logicType = LOGIC_NONE;
        currentSnap.isStart = (ctx.size() == 1);
        currentSnap.isEnd = false;
        int contextID = getContextID(currentSnap);

        int currentPosPct = min(100, (int)((float)(ctx.size() - 1) / max(1, hp.MAX_GEN_STEP) * 100));
        int posScore = calcPositionalScore(targetTid, contextID, currentPosPct);
        score += posScore;
    }

    score += calcTemplateScore(ctx, targetTid, token);

    // ========== 修改：词性硬编码加分（使用 token.getPosForToken 获取词性） ==========
    if (!ctx.empty()) {
        int last = ctx.back();
        AutoPosType lastPos = token.getPosForToken(last);
        AutoPosType curPos = token.getPosForToken(targetTid);

        // 词性组合硬编码规则
        if (lastPos == POS_NOUN && curPos == POS_AUX) {
            score += 60;
        } else if (lastPos == POS_AUX && curPos == POS_NOUN) {
            score += 50;
        } else if (lastPos == POS_PRON && curPos == POS_VERB) {
            score += 50;
        } else if (lastPos == POS_VERB && curPos == POS_NOUN) {
            score += 40;
        } else if (lastPos == POS_ADJ && curPos == POS_NOUN) {
            score += 50;
        } else if (lastPos == POS_PUNCT && curPos == POS_PUNCT) {
            score = -1;
        } else if (lastPos == POS_AUX && curPos == POS_AUX) {
            score = -1;
        } else if (lastPos == POS_PREP && curPos == POS_PREP) {
            score = -1;
        }
    }

    return max(0, score);
}
int BrainCortex::calcFluency(const vector<int>& seq,TextTokenizer& token){
    int flu=hp.FLUENCY_BASE_SCORE;
    vector<string> chars;
    for(int tid:seq){ lock_guard<recursive_mutex> lock(tokenMtx); if(tid>3 && tid>=0 && tid<(int)token.vocabList.size()) chars.push_back(token.vocabList[tid]); }
    for(int i=1;i<(int)chars.size();i++){
        string c=chars[i], p=chars[i-1];
        if(token.isPunctuation(c) && token.isPunctuation(p)) flu-=hp.PUNCT_DUPLICATE_PENALTY;
        else{
            unordered_set<string> aux={"的","是","了","也","就","都","要","在"};
            if(!token.isPunctuation(c) && c==p && !aux.count(c)){ flu-=hp.CHAR_DUPLICATE_PENALTY; if(i>=2 && chars[i-2]==c) flu-=hp.CHAR_DUPLICATE_PENALTY*2; }
        }
    }
    if(!chars.empty() && token.isPunctuation(chars.front())) flu-=hp.ABNORMAL_SEG_PENALTY;
    if(chars.size()>1 && token.isPunctuation(chars.back())) flu-=hp.ABNORMAL_SEG_PENALTY;
    return max(0,flu);
}

bool BrainCortex::checkConsistency(const vector<int>& seq, TextTokenizer& token) {
    static unordered_map<string, string> antonymMap = {
        {"高", "矮"}, {"大", "小"}, {"好", "坏"}, {"热", "冷"}, {"多", "少"}
    };
    unordered_set<string> statements;
    for (size_t i = 0; i + 2 < seq.size(); i++) {
        string s = token.decode({seq[i], seq[i+1], seq[i+2]});
        statements.insert(s);
        for (auto& entry : antonymMap) {
            const string& word = entry.first;
            const string& antonym = entry.second;
            if (s.find(word) != string::npos) {
                string opp = s;
                size_t pos = opp.find(word);
                opp.replace(pos, word.size(), antonym);
                if (statements.count(opp)) return false;
            }
        }
    }
    return true;
}

int BrainCortex::evaluateOutput(const vector<int>& seq,TextTokenizer& token,int mode){
    int base=50, valid=0, newChar=0;
    unordered_set<int> freqSet;
    if(seq.empty()) return 15;
    for(int tid:seq){ lock_guard<recursive_mutex> lock(tokenMtx); if(tid<=3 || tid<0 || tid>=(int)token.vocabList.size()) continue; freqSet.insert(tid); valid++; if(token.isNewChar(token.vocabList[tid])) newChar++; }
    if(valid>0){
        int rate=newChar*100/valid;
        if(mode==1){ if(rate>50) base-=15; } else if(mode==2){ if(rate>10) base-=30; } else { if(rate>30) base-=20; }
    }
    int flu=calcFluency(seq,token);
    int total = (mode==1)?(base*7/10+flu*3/10):((mode==2)?(base*2/10+flu*8/10):(base*5/10+flu*5/10));
    if(valid<5) total=10;
    if(!checkConsistency(seq, token)) total -= 20;
    return max(15,min(100,total));
}

bool BrainCortex::deepReflect(const vector<int>& seq, int quality, TextTokenizer& token) {
    if(quality < 25) {
        for(size_t i=0; i+1<seq.size(); i++) {
            int a = seq[i], b = seq[i+1];
            if(a<=3 || b<=3) continue;
            Neuron* nA = findNeuronByToken(a);
            if(nA) {
                DynamicEdge* e = nA->findOutput(b);
                if(e) {
                    e->weight = max(0, e->weight - 3);
                    if(e->weight == 0 && e->permanent == 0) {
                        for(auto it = nA->outputs.begin(); it != nA->outputs.end(); ++it)
                            if(it->target == b) { nA->outputs.erase(it); break; }
                    }
                }
            }
        }
        return false;
    } else if(quality > 75) {
        for(size_t i=0; i+2<seq.size(); i+=2) {
            int a = seq[i], c = seq[i+2];
            if(a<=3 || c<=3) continue;
            Neuron* nA = findNeuronByToken(a);
            if(nA && !nA->findOutput(c)) {
                nA->linkOut(c, LOGIC_PROGRESS);
            }
        }
        return true;
    }
    return false;
}

bool BrainCortex::reflect(const vector<int>& seq,TextTokenizer& token){
    if(seq.empty()) return false;
    int q=evaluateOutput(seq,token);
    int flu=calcFluency(seq,token);
    int valid=0; for(int x:seq) if(x>3) valid++;
    int boost=0,decay=0,fBoost=0,fDecay=0; bool ok=false;
    if(q>=55){ boost=valid>5?hp.REFLECT_STRENGTH*2:hp.REFLECT_STRENGTH; ok=true; }
    else if(q>=25) boost=hp.REFLECT_STRENGTH/2;
    else decay=1;
    const int FLU_GOOD=hp.FLUENCY_BASE_SCORE*7/10, FLU_BAD=hp.FLUENCY_BASE_SCORE*3/10;
    if(flu>=FLU_GOOD) fBoost=2; else if(flu<=FLU_BAD) fDecay=1;
    for(int i=0;i<(int)seq.size()-1;i++){
        int a=seq[i],b=seq[i+1]; if(a<=3 || b<=3) continue;
        lock_guard<recursive_mutex> lock(cortexMtx);
        for(auto& layer:layers) for(auto& p:layer) if(p.boundTokens.count(a)){
            DynamicEdge* e=p.findOutput(b); if(!e) continue;
            if(boost>0 || fBoost>0){ e->boostStrong(boost+fBoost); p.emotionEnergy=min(10,p.emotionEnergy+1); }
            if(decay>0) e->weight=max(0,e->weight-decay);
            if(fDecay>0) e->weight=max(0,e->weight-fDecay);
        }
    }
    deepReflect(seq, q, token);
    updateAllLayers(); pruneWeakNodes();
    return ok;
}

void BrainCortex::setGoal(int g){ goalIntent=g; }
void BrainCortex::setStyle(StyleIntent s){ curStyle=s; }
int BrainCortex::totalNeurons(){ lock_guard<recursive_mutex> lock(cortexMtx); int sum=0; for(auto& l:layers) sum+=l.size(); return sum; }

void BrainCortex::addRandomNeuron() {
    lock_guard<recursive_mutex> lock(cortexMtx);
    if (layers.empty()) return;
    int bestLayer = 0;
    float minAvgAct = 1e9;
    for(int l=0; l<(int)layers.size(); l++){
        float sumAct=0;
        for(auto& neu : layers[l]) sumAct += neu.activation;
        float avg = layers[l].empty() ? 0 : sumAct/layers[l].size();
        if(avg < minAvgAct){
            minAvgAct = avg;
            bestLayer = l;
        }
    }
    int nid = createNeuron(bestLayer);
    Neuron* neu = findNeuron(bestLayer, nid);
    if(!neu) return;
    int numConn = rand() % 3 + 1;
    for(int i=0; i<numConn && i<(int)layers[bestLayer].size()-1; i++){
        int targetIdx = rand() % layers[bestLayer].size();
        if(layers[bestLayer][targetIdx].neuronId == nid) continue;
        if(neu->canConnect(layers[bestLayer][targetIdx])) {
            neu->linkOut(layers[bestLayer][targetIdx].neuronId);
            layers[bestLayer][targetIdx].linkIn(nid);
        }
    }
    cout << "[可变拓扑] 新增神经元 " << nid << " 在第 " << bestLayer << " 层" << endl;
    rebuildLayerIndex();
}

void BrainCortex::maybeSplitLayer() {
    lock_guard<recursive_mutex> lock(cortexMtx);
    for (int l=0; l<(int)layers.size(); l++) {
        if (layers[l].size() > hp.MAX_NEURONS_PER_LAYER && layers.size() < hp.CORTEX_LAYERS+2) {
            int newLayerIdx = layers.size();
            layers.emplace_back();
            layerIdToIndex.emplace_back();
            int mid = layers[l].size() / 2;
            for (int i=mid; i<(int)layers[l].size(); i++) {
                Neuron neu = layers[l][i];
                neu.layer = newLayerIdx;
                layers[newLayerIdx].push_back(neu);
            }
            layers[l].erase(layers[l].begin()+mid, layers[l].end());
            cout << "[可变拓扑] 层 " << l << " 分裂，新增层 " << newLayerIdx << endl;
            for (auto& fromNeu : layers[l]) {
                for (int k=0; k<3; k++) {
                    int targetIdx = rand() % layers[newLayerIdx].size();
                    if (fromNeu.canConnect(layers[newLayerIdx][targetIdx])) {
                        fromNeu.linkOut(layers[newLayerIdx][targetIdx].neuronId);
                    }
                }
            }
            rebuildLayerIndex();
            break;
        }
    }
}

void BrainCortex::rewiring() {
    lock_guard<recursive_mutex> lock(cortexMtx);
    int rewired = 0;
    for (auto& layer : layers) {
        for (auto& neu : layer) {
            if (neu.outputs.empty()) continue;
            int idx = -1;
            float minImp = 1e9;
            for (int i=0; i<(int)neu.outputs.size(); i++) {
                float imp = neu.outputs[i].importance();
                if (imp < minImp) {
                    minImp = imp;
                    idx = i;
                }
            }
            if (idx >=0 && minImp < 5.0f) {
                int oldTarget = neu.outputs[idx].target;
                vector<int> candidates;
                for (auto& other : layer) {
                    if (other.neuronId != neu.neuronId && other.neuronId != oldTarget && neu.canConnect(other))
                        candidates.push_back(other.neuronId);
                }
                if (!candidates.empty()) {
                    int newTarget = candidates[rand() % candidates.size()];
                    neu.outputs[idx].target = newTarget;
                    neu.outputs[idx].weight = 1;
                    neu.outputs[idx].lifeCycle = hp.EDGE_DECAY_STEP;
                    rewired++;
                    if (rewired >= hp.REWIRING_EDGES) break;
                }
            }
        }
        if (rewired >= hp.REWIRING_EDGES) break;
    }
    if (rewired>0) cout << "[可变拓扑] 重配了 " << rewired << " 条低重要性边" << endl;
}

void BrainCortex::maybeMutateTopology(int lastQuality) {
    generationCounter++;
    if (lastQuality < hp.NEUROGENESIS_QUALITY_THRESH && neurogenesisCooldown <= 0) {
        addRandomNeuron();
        neurogenesisCooldown = hp.NEUROGENESIS_COOLDOWN;
    }
    if (neurogenesisCooldown > 0) neurogenesisCooldown--;
    if (generationCounter % 10 == 0) maybeSplitLayer();
    if (generationCounter % hp.REWIRING_INTERVAL == 0) rewiring();
}
// ========== 自我模型实现 ==========
void BrainCortex::initSelfModel() {
    selfNeurons.clear();
    auto add = [this](const string& name, float initVal=0.0f) {
        SelfNeuron sn{name, initVal, initVal, 0.0f};
        selfNeurons.push_back(sn);
    };
    add("AvgActivation");
    add("QualitySliding");
    add("EmotionArousal");
    add("InferenceConfidence");
    add("TopoChangeRate");
    add("ShortMemNovelty");
    add("PredictionErrorGlobal");
    add("SurpriseAvg");          // ========== 新增 ==========
    add("ErrorTrend");           // ========== 新增 ==========
    while((int)selfNeurons.size() < hp.SELF_NEURON_COUNT) add("Extra");
}

void BrainCortex::setSelfValue(const string& name, float val) {
    for(auto& sn : selfNeurons) if(sn.name == name) { sn.value = val; return; }
}

float BrainCortex::getSelfValue(const string& name) const {
	lock_guard<recursive_mutex> lock(cortexMtx); 
    for(const auto& sn : selfNeurons) if(sn.name == name) return sn.value;
    return 0.0f;
}

void BrainCortex::updateSelfModel(int lastQuality, float inferenceConf, float topoChangeRate) {
    float totalAct = 0;
    int totalNeurons = 0;
    for(auto& layer : layers) for(auto& neu : layer) { totalAct += neu.activation; totalNeurons++; }
    float avgAct = (totalNeurons>0) ? (totalAct/totalNeurons) : 0;
    setSelfValue("AvgActivation", min(100.0f, avgAct));
    qualityHistoryForSelf.push_back(lastQuality);
    if(qualityHistoryForSelf.size() > 10) qualityHistoryForSelf.pop_front();
    float avgQual = 0;
    for(int q : qualityHistoryForSelf) avgQual += q;
    avgQual /= qualityHistoryForSelf.size();
    setSelfValue("QualitySliding", avgQual);
    setSelfValue("EmotionArousal", (float)curEmo.intensity);
    setSelfValue("InferenceConfidence", inferenceConf * 100.0f);
    setSelfValue("TopoChangeRate", topoChangeRate * 100.0f);
    setSelfValue("PredictionErrorGlobal", avgPredictionError);
    setSelfValue("SurpriseAvg", avgSurprise);
    
    // 计算误差趋势
    if (predictionHistory.size() > 10) {
        float recent = 0, old = 0;
        int n = predictionHistory.size();
        for (int i = n-5; i < n; i++) recent += predictionHistory[i].surprise;
        for (int i = 0; i < 5; i++) old += predictionHistory[i].surprise;
        setSelfValue("ErrorTrend", (recent - old) / 5.0f);
    }
    
    unordered_set<int> unique(shortMemory.begin(), shortMemory.end());
    float novelty = shortMemory.empty() ? 0 : (float)unique.size() / shortMemory.size() * 100.0f;
    setSelfValue("ShortMemNovelty", novelty);
    float sumErr = 0;
    for(auto& sn : selfNeurons) {
        float oldPred = sn.predicted;
        sn.predicted = hp.SELF_PREDICTION_ALPHA * sn.value + (1 - hp.SELF_PREDICTION_ALPHA) * sn.predicted;
        sn.error = fabs(sn.value - oldPred);
        sumErr += sn.error;
    }
    avgPredictionError = sumErr / selfNeurons.size();
    setSelfValue("PredictionErrorGlobal", avgPredictionError);
}

// ========== 两环控制器 ==========
void BrainCortex::updateTwoLoopController(int quality) {
    T_history.push_back(T); if (T_history.size()>5) T_history.pop_front();
    E_history.push_back(E); if (E_history.size()>5) E_history.pop_front();
    R_history.push_back(R); if (R_history.size()>5) R_history.pop_front();
    int T_avg=0, E_avg=0, R_avg=0;
    for (int v : T_history) T_avg += v; T_avg /= T_history.size();
    for (int v : E_history) E_avg += v; E_avg /= E_history.size();
    for (int v : R_history) R_avg += v; R_avg /= R_history.size();

    int newT = (alpha1 * T * (100 - T)) / 100 + (beta1 * E) / 2 + (gamma1 * M1) / 10;
    int newE = (alpha2 * E * (100 - E)) / 100 + (beta2 * T) / 3 + (gamma2 * M2) / 10;
    int newR = (alpha3 * R * (100 - R)) / 100 + (beta3 * (T - E)) / 2 + (gamma3 * M1 * M2) / 100;
    T = max(10, min(90, newT));
    E = max(5, min(60, newE));
    R = max(3, min(25, newR));

    int qualityInput = max(0, min(100, quality));
    int newM1 = (delta1 * M1 * (100 - M1)) / 100 + (epsilon1 * (T_avg - E_avg)) / 2 + (zeta1 * qualityInput) / 10;
    int newM2 = (delta2 * M2 * (100 - M2)) / 100 + (epsilon2 * (R_avg - 50)) / 2 + (zeta2 * (100 - qualityInput)) / 10;
    M1 = max(10, min(90, newM1));
    M2 = max(10, min(90, newM2));

    hp.GEN_TEMP = 20 + (T * 70) / 100;
    hp.CREATE_CHAR_RATE_NORMAL = 10 + (E * 20) / 100;
    hp.REFLECT_STRENGTH = 5 + (R * 15) / 100;
}

string BrainCortex::generateConsciousReport() {
    string report;
    float avgAct = getSelfValue("AvgActivation");
    float quality = getSelfValue("QualitySliding");
    float emotion = getSelfValue("EmotionArousal");
    float predErr = getSelfValue("PredictionErrorGlobal");
    float surprise = getSelfValue("SurpriseAvg");
    float novelty = getSelfValue("ShortMemNovelty");
    float errorTrend = getSelfValue("ErrorTrend");
    
    if(avgAct < 15.0f) report += "我感到皮层沉寂，思维不活跃。";
    else if(avgAct > 60.0f) report += "我感到思维高度活跃。";
    
    if(quality < 30.0f) report += "我对自己最近的生成质量很不满意。";
    else if(quality > 70.0f) report += "我对自己最近的输出感到满意。";
    
    if(emotion > 70.0f) report += "我情绪高涨。";
    else if(emotion < 20.0f) report += "我情绪低落。";
    
    if(predErr > hp.PREDICTION_ERROR_THRESH) report += "我对未来的预测很不确定，内心困惑。";
    
    // ========== 新增：预测误差报告 ==========
    if(surprise > 4.0f) report += "我经常感到意外，世界比我想象的复杂。";
    else if(surprise < 0.3f) report += "一切都太可预测了，有些无聊。";
    
    if(errorTrend > 2.0f) report += "我注意到预测误差在上升，需要调整认知。";
    else if(errorTrend < -2.0f) report += "我对世界的理解越来越准确了。";
    
    if(novelty > 60.0f) report += "我注意到短时记忆中出现了很多新奇的词汇。";
    if(report.empty()) report = "我感觉平静，一切正常。";
    return report;
}
bool BrainCortex::hasGoal(const string& desc) {
    for(auto& g : innerGoals) if(g.description == desc && g.active) return true;
    return false;
}

void BrainCortex::generateInnerGoals() {
    innerGoals.erase(remove_if(innerGoals.begin(), innerGoals.end(),
        [this](const InnerGoal& g){ return !g.active || (stepCounter - g.assignedStep > 50); }),
        innerGoals.end());
    if(innerGoals.size() >= 3) return;
    float predErr = getSelfValue("PredictionErrorGlobal");
    float novelty = getSelfValue("ShortMemNovelty");
    float quality = getSelfValue("QualitySliding");
    if(predErr > 40.0f && !hasGoal("reduce_prediction_error")) {
        innerGoals.push_back({"reduce_prediction_error", -1, 0.0f, stepCounter, true});
        cout << "[意识] 我设定目标：降低预测误差。" << endl;
    }
    if(quality < 40.0f && !hasGoal("improve_quality")) {
        innerGoals.push_back({"improve_quality", -1, 0.0f, stepCounter, true});
        cout << "[意识] 我设定目标：提高输出质量。" << endl;
    }
    if(novelty < 30.0f && !hasGoal("explore_novelty")) {
        innerGoals.push_back({"explore_novelty", -1, 0.0f, stepCounter, true});
        cout << "[意识] 我设定目标：增加词汇新颖度。" << endl;
    }
}

void BrainCortex::consciousBroadcast() {
    string report = generateConsciousReport();
    if(report != lastConsciousReport && (stepCounter % hp.CONSCIOUSNESS_REPORT_INTERVAL == 0)) {
        lastConsciousReport = report;
        cout << "[意识内省] " << report << endl;
    }
}

// ========== 注意力子图 ==========
void BrainCortex::buildAttentionSubgraph(const vector<int>& ctx, TextTokenizer& token) {
    lock_guard<recursive_mutex> lock(cortexMtx);
    curSubgraph.clear();
    unordered_set<int> seedTokens;
    int start = max(0, (int)ctx.size() - attentionWindowSize);
    for (int i = start; i < (int)ctx.size(); ++i) {
        if (ctx[i] > 3) seedTokens.insert(ctx[i]);
    }
    for (int t : shortMemory) if (t > 3) seedTokens.insert(t);
    for (int t : logicAnchorTokens) if (t > 3) seedTokens.insert(t);
    for (int tid : seedTokens) {
        Neuron* neu = findNeuronByToken(tid);
        if (!neu) continue;
        curSubgraph.neurons.insert(neu->neuronId);
        for (auto& e : neu->outputs) {
            if (curSubgraph.neurons.size() >= maxSubgraphNeurons) goto finish;
            curSubgraph.neurons.insert(e.target);
        }
        for (auto& e : neu->inputs) {
            if (curSubgraph.neurons.size() >= maxSubgraphNeurons) goto finish;
            curSubgraph.neurons.insert(e.target);
        }
    }
    finish:
    for (int nid : curSubgraph.neurons) {
        Neuron* neu = findNeuronById(nid);
        if (!neu) continue;
        for (int tid : neu->boundTokens) {
            curSubgraph.tokens.insert(tid);
            if (curSubgraph.tokens.size() >= maxSubgraphTokens) break;
        }
    }
    for (int t : seedTokens) curSubgraph.tokens.insert(t);
}

int BrainCortex::calcScoreSubgraph(const vector<int>& ctx, int targetTid, TextTokenizer& token) {
    if (!curSubgraph.tokens.count(targetTid)) return 0;
    return calcScore(ctx, targetTid, token);
}

void BrainCortex::offlineConsolidation(TextTokenizer& token, int maxSeconds) {
    cout << "[休眠] 进入离线巩固模式..." << endl;
    auto startTime = chrono::steady_clock::now();
    int iteration = 0;
    while (true) {
        this_thread::sleep_for(chrono::seconds(10));
        if (userInputWaiting) {
            cout << "[休眠] 检测到用户输入，立即唤醒。" << endl;
            break;
        }
        auto elapsed = chrono::duration_cast<chrono::seconds>(
            chrono::steady_clock::now() - startTime).count();
        if (elapsed >= maxSeconds) {
            cout << "[休眠] 达到最大巩固时间，退出。" << endl;
            break;
        }
        
        lock_guard<recursive_mutex> lock(cortexMtx);
        switch (iteration % 4) {
            case 0:
                // 压缩情节记忆
                if (episodicMemory.size() > 100) {
                    vector<Episode> kept;
                    for (auto& ep : episodicMemory) {
                        if (ep.importance > 0.5f) kept.push_back(ep);
                        if (kept.size() >= 50) break;
                    }
                    episodicMemory = deque<Episode>(kept.begin(), kept.end());
                    cout << "[休眠] 情节记忆压缩至 " << episodicMemory.size() << " 条。" << endl;
                }
                break;
            case 1:
                pruneWeakNodes();
                updateAllLayers();
                cout << "[休眠] 执行了弱连接修剪。" << endl;
                break;
            case 2:
                autoInduceRelationsLocked(token);
                cout << "[休眠] 自动归纳逻辑关系。" << endl;
                break;
            case 3:
                // ========== 新增：全局误差整合 ==========
                integrateGlobalPredictionError();
                cout << "[休眠] 整合全局预测误差，avgSurprise=" << avgSurprise << endl;
                break;
        }
        iteration++;
    }
    curSubgraph.clear();
}

// ========== 模板类比和评分 ==========
int BrainCortex::calcTemplateScore(const vector<int>& ctx, int candidate, TextTokenizer& token) {
    if (ctx.empty()) return 0;
    int score = 0;
    int last = ctx.back();
    int prev = (ctx.size()>=2) ? ctx[ctx.size()-2] : -1;
    vector<int> matched;
    if (prev != -1) {
        matched = token.matchPhraseByPrefix(prev, last);
    } else {
        matched = token.matchPhraseByPrefix(last);
    }
    for (int t : matched) {
        if (t == candidate) {
            for (auto& pt : token.phraseTemplates) {
                if (pt.first.size() >= 2 && pt.first[pt.first.size()-1] == t) {
                    score += pt.second * 3;
                    break;
                }
            }
        }
    }
    vector<AutoPosType> recentPos;
    for (int i = max(0, (int)ctx.size()-2); i < (int)ctx.size(); ++i) {
        Neuron* n = findNeuronByToken(ctx[i]);
        recentPos.push_back(n ? n->autoPos : POS_UNKNOWN);
    }
    AutoPosType candPos = POS_UNKNOWN;
    Neuron* cn = findNeuronByToken(candidate);
    if (cn) candPos = cn->autoPos;
    AutoPosType expected = token.matchPosSkeleton(recentPos);
    if (expected != POS_UNKNOWN && candPos == expected) {
        score += 40;
    }
    return score;
}

vector<int> BrainCortex::analogicalGenerate(const vector<int>& context, TextTokenizer& token) {
    if (context.empty()) return {};
    vector<AutoPosType> recentPos;
    int start = max(0, (int)context.size() - 3);
    for (int i = start; i < (int)context.size(); ++i) {
        int tid = context[i];
        Neuron* n = findNeuronByToken(tid);
        recentPos.push_back(n ? n->autoPos : POS_UNKNOWN);
    }
    AutoPosType expectedPos = token.matchPosSkeleton(recentPos);
    if (expectedPos == POS_UNKNOWN) return {};
    int fillToken = -1;
    for (int i = context.size()-1; i >= 0; --i) {
        int t = context[i];
        Neuron* n = findNeuronByToken(t);
        if (n && (n->autoPos == POS_NOUN || n->autoPos == POS_VERB || n->autoPos == POS_ADJ)) {
            fillToken = t;
            break;
        }
    }
    if (fillToken == -1) return {};
    int lastToken = context.back();
    vector<int> candidates = token.matchPhraseByPrefix(lastToken);
    if (candidates.empty()) return {};
    int chosen = candidates[rand() % candidates.size()];
    return {chosen};
}

// ============================================================
//  四核架构：三个评分核 + 元核
// ============================================================

// ============================================================
//  1. 逻辑核：包含逻辑关系、词性、目标意图、逻辑锚点、一致性
// ============================================================
std::unordered_map<int, int> BrainCortex::calcLogicScore(const std::vector<int>& ctx, TextTokenizer& token) {
	lock_guard<recursive_mutex> lock(cortexMtx);
    std::unordered_map<int, int> scores;
    if (ctx.empty()) return scores;
    int last = ctx.back();
    Neuron* neu = findNeuronByToken(last);
    if (!neu) return scores;

    // 基础逻辑边
    for (auto& edge : neu->outputs) {
        int tid = edge.target;
        if (tid <= 3) continue;
        int sc = edge.total() * 2;
        if (edge.logic != LOGIC_NONE) sc += 30;
        if (isLogicalConsistent(last, tid, edge.logic, ctx, token)) sc += 10;

        // ---------- 语境匹配加分 ----------
        ContextSnapshot snap;
        if (!ctx.empty()) {
            snap.prevToken = (ctx.size() >= 2) ? ctx[ctx.size()-2] : -1;
            snap.nextToken = -1;
            snap.prevPos = (snap.prevToken >= 0 && snap.prevToken < (int)token.tokenPosCache.size()) ?
                           token.tokenPosCache[snap.prevToken] : POS_UNKNOWN;
            snap.nextPos = POS_UNKNOWN;
            snap.logicType = edge.logic;
            snap.isStart = (ctx.size() == 1);
            snap.isEnd = false;
        } else {
            snap.prevPos = POS_UNKNOWN;
            snap.nextPos = POS_UNKNOWN;
            snap.logicType = LOGIC_NONE;
            snap.isStart = true;
            snap.isEnd = false;
        }
        sc += edge.getContextWeight(snap) / 3;

        scores[tid] += sc;
    }

    // 词性约束、目标、锚点
        // ---------- 词性组合硬编码（完整 8 条，含 NOUN+AUX 和 AUX+NOUN） ----------
    AutoPosType lastPos = (last >= 0 && last < (int)token.tokenPosCache.size()) ? token.tokenPosCache[last] : POS_UNKNOWN;
    for (auto& kv : scores) {
        int tid = kv.first;
        AutoPosType curPos = (tid >= 0 && tid < (int)token.tokenPosCache.size()) ? token.tokenPosCache[tid] : POS_UNKNOWN;
        if (lastPos == POS_NOUN && curPos == POS_AUX) scores[tid] += 60;
        else if (lastPos == POS_AUX && curPos == POS_NOUN) scores[tid] += 50;
        else if (lastPos == POS_PRON && curPos == POS_VERB) scores[tid] += 50;
        else if (lastPos == POS_VERB && curPos == POS_NOUN) scores[tid] += 40;
        else if (lastPos == POS_ADJ && curPos == POS_NOUN) scores[tid] += 50;
        else if (lastPos == POS_PUNCT && curPos == POS_PUNCT) scores[tid] = -1;
        else if (lastPos == POS_AUX && curPos == POS_AUX) scores[tid] = -1;
        else if (lastPos == POS_PREP && curPos == POS_PREP) scores[tid] = -1;
    }

    if (goalIntent != -1 && scores.count(goalIntent)) scores[goalIntent] *= 2;
    if (!logicAnchorTokens.empty()) {
        int anchor = logicAnchorTokens.back();
        if (scores.count(anchor)) scores[anchor] += 30;
    }

    return scores;
}

// ============================================================
//  2. 联想核：包含激活扩散、嵌入相似度、全局工作空间、长期记忆、知识概念
// ============================================================
std::unordered_map<int, int> BrainCortex::calcAssocScore(const std::vector<int>& ctx, TextTokenizer& token) {
	lock_guard<recursive_mutex> lock(cortexMtx);
    std::unordered_map<int, int> scores;

    // 激活传播
    for (int nid : curSubgraph.neurons) {
        Neuron* neu = findNeuronById(nid);
        if (!neu || neu->activation == 0) continue;
        for (auto& edge : neu->outputs) {
            int tid = edge.target;
            if (tid <= 3) continue;

            int sc = edge.total() * neu->activation / 5;

            // 语境匹配
            ContextSnapshot snap;
            if (!ctx.empty()) {
                int curTok = ctx.back();
                snap.prevToken = (ctx.size() >= 2) ? ctx[ctx.size()-2] : -1;
                snap.nextToken = -1;
                snap.prevPos = (snap.prevToken >= 0 && snap.prevToken < (int)token.tokenPosCache.size()) ?
                               token.tokenPosCache[snap.prevToken] : POS_UNKNOWN;
                snap.nextPos = POS_UNKNOWN;
                snap.logicType = edge.logic;
                snap.isStart = (ctx.size() == 1);
                snap.isEnd = false;
            } else {
                snap.prevPos = POS_UNKNOWN;
                snap.nextPos = POS_UNKNOWN;
                snap.logicType = LOGIC_NONE;
                snap.isStart = true;
                snap.isEnd = false;
            }
            sc += edge.getContextWeight(snap) / 2;

            if (std::find(ctx.begin(), ctx.end(), tid) != ctx.end()) sc /= 2;
            scores[tid] += sc;
        }
    }

    // globalAttention
    int start = max(0, (int)ctx.size() - hp.ATTENTION_SPAN);
    for (int i = start; i < (int)ctx.size(); ++i) {
        int t = ctx[i];
        int pos = i - start + 1;
        Neuron* neu = findNeuronByToken(t);
        if (!neu) continue;
        for (auto& edge : neu->outputs) {
            int tid = edge.target;
            if (tid <= 3) continue;
            int add = edge.total() * pos;
            if (edge.logic != LOGIC_NONE) add += 3;
            scores[tid] += add;
        }
    }

    // 短时记忆边遍历
    for (int t : shortMemory) {
        for (auto& layer : layers) {
            for (auto& p : layer) {
                if (p.boundTokens.count(t)) {
                    for (auto& e : p.outputs) {
                        int tid = e.target;
                        if (tid <= 3) continue;
                        scores[tid] += e.total();
                    }
                }
            }
        }
    }

    // 情感能量 + 成熟度
    for (auto& layer : layers) {
        for (auto& p : layer) {
            for (int tid : p.boundTokens) {
                if (tid <= 3) continue;
                int add = (p.emotionEnergy + curEmo.intensity) / 3 + p.maturity / 4;
                scores[tid] += add;
            }
        }
    }

    // 全局工作空间
    vector<int> topWorkspace = workspace.getTop(3);
    for (int tid : topWorkspace) {
        if (scores.count(tid)) scores[tid] += 20;
        else scores[tid] = 20;
    }

    // 长期文本记忆
    for (auto& seg : longTextMemory) {
        for (int t : seg) {
            if (scores.count(t)) scores[t] += 2;
            else scores[t] = 2;
        }
    }
    for (int t : longTermMemory) {
        if (scores.count(t)) scores[t] += 1;
        else scores[t] = 1;
    }

    // 知识概念
    for (int cid : knowledgeConcepts) {
        if (scores.count(cid)) scores[cid] += 15;
        else scores[cid] = 15;
    }

    // 嵌入相似度
    if (!ctx.empty()) {
        std::vector<float> ctxEmb(token.EMBED_DIM, 0.0f);
        int cnt = 0;
        for (int t : ctx) {
            if (token.tokenEmbedding.find(t) != token.tokenEmbedding.end()) {
                for (int i = 0; i < token.EMBED_DIM; ++i) ctxEmb[i] += token.tokenEmbedding[t][i];
                cnt++;
            }
        }
        if (cnt > 0) {
            for (int i = 0; i < token.EMBED_DIM; ++i) ctxEmb[i] /= cnt;
            std::vector<int> keys;
            for (auto& kv : scores) keys.push_back(kv.first);
            for (int tid : keys) {
                if (token.tokenEmbedding.find(tid) != token.tokenEmbedding.end()) {
                    float sim = token.cosineSimilarity(ctxEmb, token.tokenEmbedding[tid]);
                    scores[tid] += (int)(sim * 20);
                }
            }
        }
    }

    // 风格调制
    if (curStyle == STYLE_EMOTION) {
        for (auto& kv : scores) kv.second = kv.second * 12 / 10;
    } else if (curStyle == STYLE_SIMPLE) {
        for (auto& kv : scores) kv.second = kv.second * 8 / 10;
    }

    return scores;
}

// ============================================================
//  3. 奖惩核：包含流畅度变化、情感能量、神经元成熟度、模板匹配
// ============================================================
std::unordered_map<int, int> BrainCortex::calcRewardScore(const std::vector<int>& ctx, TextTokenizer& token) {
	lock_guard<recursive_mutex> lock(cortexMtx);
    std::unordered_map<int, int> scores;

    // TD 差分
    if (stepFluency.size() >= 2) {
        int currentFlu = calcFluency(ctx, token);
        int prevFlu = stepFluency.back();
        int delta = currentFlu - prevFlu;
        if (ctx.empty()) return scores;
        int last = ctx.back();
        Neuron* neu = findNeuronByToken(last);
        if (neu) {
            for (auto& edge : neu->outputs) {
                int tid = edge.target;
                if (tid <= 3) continue;
                int adjust = delta / 2;
                if (adjust != 0) scores[tid] += adjust;
            }
        }
    }

    // 情感因子（用于温度调节）
    float emotionFactor = (curEmo.intensity > 60) ? 1.5f : (curEmo.intensity < 5 ? 0.6f : 1.0f);

    // 模板评分
    if (!ctx.empty()) {
        int last = ctx.back();
        Neuron* neu = findNeuronByToken(last);
        if (neu) {
            for (auto& edge : neu->outputs) {
                int tid = edge.target;
                if (tid <= 3) continue;
                int tmpl = calcTemplateScore(ctx, tid, token);
                if (tmpl > 0) scores[tid] += tmpl;
            }
        }
    }

    return scores;
}

// ============================================================
//  新的四核 singleGenerate
// ============================================================
vector<int> BrainCortex::singleGenerate(vector<int> ctx, TextTokenizer& token) {
    logger.logDebug("singleGenerate 开始, ctx.size=" + to_string(ctx.size()));
    
    buildAttentionSubgraph(ctx, token);
    logger.logDebug("buildAttentionSubgraph 完成");
    
    layerForwardPass(token);
    logger.logDebug("layerForwardPass 完成");
    
    for (auto& lay : layers) {
        for (auto& neu : lay) {
            neu.activate();
        }
    }
    logger.logDebug("activate 完成");
    
    clearStepRecords();
    logger.logDebug("clearStepRecords 完成");

    unordered_set<string> aux = { "的","一","是","了","也","就","都","要","在","和","有","这","那" };

    int step = 0;
    for (; step < hp.MAX_GEN_STEP; ++step) {
        logger.logDebug("主循环第 " + to_string(step) + " 步开始");
        
        updateGains(ctx, token);
        logger.logDebug("updateGains 完成");

        // ============================================================
        // 1. 三核并行评分
        // ============================================================
        logger.logDebug("开始并行评分...");
		// 改成：
		std::unordered_map<int, int> logicScores, assocScores, rewardScores;
		#pragma omp parallel sections
		{
		    #pragma omp section
		    {
		        #pragma omp critical
		        {
		            logicScores = calcLogicScore(ctx, token);
		        }
		    }
		    #pragma omp section
		    {
		        #pragma omp critical
		        {
		            assocScores = calcAssocScore(ctx, token);
		        }
		    }
		    #pragma omp section
		    {
		        #pragma omp critical
		        {
		            rewardScores = calcRewardScore(ctx, token);
		        }
		    }
		}
        logger.logDebug("评分完成, logic=" + to_string(logicScores.size()) + 
                        ", assoc=" + to_string(assocScores.size()) +
                        ", reward=" + to_string(rewardScores.size()));

        // ============================================================
        // 2. 交叉验证
        // ============================================================
        auto getTopK = [](const std::unordered_map<int, int>& scores, int k) {
            std::vector<std::pair<int, int>> vec(scores.begin(), scores.end());
            std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::vector<int> result;
            for (int i = 0; i < std::min(k, (int)vec.size()); ++i) result.push_back(vec[i].first);
            return result;
        };
        std::vector<int> topL = getTopK(logicScores, 10);
        std::vector<int> topA = getTopK(assocScores, 10);
        std::vector<int> topR = getTopK(rewardScores, 10);
        logger.logDebug("getTopK 完成");

        auto crossValidate = [](std::unordered_map<int, int>& self,
                                const std::unordered_map<int, int>& other,
                                const std::vector<int>& otherTop,
                                float weight) {
            for (auto& kv : self) {
                int tid = kv.first;
                if (other.count(tid)) {
                    kv.second += (int)(other.at(tid) * weight);
                } else {
                    kv.second = kv.second * 9 / 10;
                }
            }
            for (int tid : otherTop) {
                if (!self.count(tid)) {
                    self[tid] = (int)(other.at(tid) * 0.5f);
                }
            }
        };

        crossValidate(logicScores, assocScores, topA, 0.3f);
        crossValidate(logicScores, rewardScores, topR, 0.2f);
        crossValidate(assocScores, logicScores, topL, 0.4f);
        crossValidate(assocScores, rewardScores, topR, 0.2f);
        crossValidate(rewardScores, logicScores, topL, 0.5f);
        crossValidate(rewardScores, assocScores, topA, 0.3f);
        logger.logDebug("交叉验证完成");

        // ============================================================
        // 3. 融合所有候选
        // ============================================================
        std::unordered_set<int> allTokens;
        for (auto& kv : logicScores) allTokens.insert(kv.first);
        for (auto& kv : assocScores) allTokens.insert(kv.first);
        for (auto& kv : rewardScores) allTokens.insert(kv.first);

        std::vector<std::pair<int, int>> candAll;
        unordered_set<int> ctxFreq(ctx.begin(), ctx.end());

        for (int tid : allTokens) {
            int sc = 0;
            if (logicScores.count(tid)) sc += logicScores[tid];
            if (assocScores.count(tid)) sc += assocScores[tid];
            if (rewardScores.count(tid)) sc += rewardScores[tid];

            if (ctxFreq.count(tid)) {
                if ((int)ctxFreq.size() >= 15) continue;
                else if ((int)ctxFreq.size() >= 10) sc /= 2;
                else if ((int)ctxFreq.size() >= 5) sc = sc * 3 / 4;
            }

            if (tid >= 0 && tid < (int)token.vocabList.size()) {
                string cw = token.vocabList[tid];
                if (aux.count(cw)) sc -= 1;
            }

            if (sc > 0) {
                candAll.emplace_back(tid, sc);
            }
        }
        logger.logDebug("融合完成, candAll.size=" + to_string(candAll.size()));

        // ============================================================
		// 4. 死锁处理
		// ============================================================
		if (candAll.empty()) {
		    logger.logDebug("candAll 为空，进入死锁处理");
		    
		    // 尝试1：类比生成
		    vector<int> analog = analogicalGenerate(ctx, token);
		    if (!analog.empty()) {
		        for (int tid : analog) {
		            int sc = calcScore(ctx, tid, token);
		            if (sc > 0) {
		                candAll.emplace_back(tid, sc);
		            }
		        }
		    }
		    
		    // 尝试2：如果类比失败，用随机候选
		    if (candAll.empty()) {
		        deadlockCounter++;
		        
		        // 如果连续死锁太多，强制退出
		        if (deadlockCounter >= DEADLOCK_THRESHOLD) {
		            logger.logDebug("死锁达到阈值，退出");
		            break;
		        }
		        
		        // 调整参数（尝试打破僵局）
		        alpha_L += 0.15f;
		        alpha_A += 0.10f;
		        alpha_R -= 0.10f;
		        
		        // 生成随机候选
		        int randomTid = -1;
		        if (token.vocabList.size() > 4) {
		            int attempts = 0;
		            while (attempts < 100) {
		                randomTid = 4 + (rand() % (token.vocabList.size() - 4));
		                // 避免选到已经在上下文中的 token
		                if (randomTid > 3 && !ctxFreq.count(randomTid)) {
		                    break;
		                }
		                attempts++;
		            }
		            // 如果 100 次都找不到不重复的，随便选一个
		            if (attempts >= 100) {
		                randomTid = 4 + (rand() % (token.vocabList.size() - 4));
		            }
		        } else if (token.vocabList.size() > 0) {
		            randomTid = rand() % token.vocabList.size();
		        }
		        
		        if (randomTid > 3 && randomTid < (int)token.vocabList.size()) {
		            int sc = calcScore(ctx, randomTid, token) + 10;
		            candAll.emplace_back(randomTid, sc);
		            logger.logDebug("随机候选生成: " + token.vocabList[randomTid]);
		        }
		    }
		}
		
		// 如果 candAll 仍然为空，说明系统彻底无法生成
		if (candAll.empty()) {
		    logger.logError("无法生成任何候选，强制结束本轮生成");
		    break;
		}
		
		// 只有在成功生成候选时才重置死锁计数器
		deadlockCounter = 0;
		logger.logDebug("死锁处理完成，候选数: " + to_string(candAll.size()));
        // ============================================================
        // 5. 新词创造
        // ============================================================
        int createRate = (curEmo.type == EMO_EXCITE) ? hp.CREATE_CHAR_RATE_EXCITE : hp.CREATE_CHAR_RATE_NORMAL;
        if (rand() % createRate == 0) {
            string nw = token.composeNewChar();
            if (!nw.empty()) {
                lock_guard<recursive_mutex> lock(tokenMtx);
                int newId = token.getTokenId(nw);
                if (newId == -1) {
                    token.vocabList.push_back(nw);
                    newId = (int)token.vocabList.size() - 1;
                    if (token.tokenEmbedding.find(newId) == token.tokenEmbedding.end()) {
                        token.tokenEmbedding[newId] = vector<float>(token.EMBED_DIM, 0.01f);
                    }
                    token.tokenPosCache.push_back(POS_UNKNOWN);
                    token.tokenIsPunctCache.push_back(token.symbolFilter.count(nw));
                    token.tokenIsNewCharCache.push_back(token.isNewChar(nw));
                }
                int sc = calcScore(ctx, newId, token) + (rand() % 2 + 1);
                candAll.emplace_back(newId, sc);
            }
        }
        logger.logDebug("新词创造完成");
		
		// 在 candAll 构建完成后，排序之前
		if (userModel.hasData()) {
		    vector<int> candidateTokens;
		    for (auto& p : candAll) candidateTokens.push_back(p.first);
		    auto userScores = userModel.simulateCandidates(candidateTokens, ctx);
		    unordered_map<int, float> userMap;
		    for (auto& us : userScores) userMap[us.first] = us.second;
		    for (auto& p : candAll) {
		        auto it = userMap.find(p.first);
		        if (it != userMap.end()) {
		            p.second += (int)(it->second * 60);  // 权重 60 分，可调
		        }
		    }
		}
        // ============================================================
        // 6. 排序 + 概率选择
        // ============================================================
        std::sort(candAll.begin(), candAll.end(),
            [](const std::pair<int,int>& a, const std::pair<int,int>& b) { return a.second > b.second; });
        if ((int)candAll.size() > hp.TOP_K_CAND) candAll.resize(hp.TOP_K_CAND);

        int temp = (curEmo.type == EMO_EXCITE) ? 48 : hp.GEN_TEMP;
        std::vector<std::pair<int,int>> prob;
        int total = 0;
        for (auto& p : candAll) {
            int w = max(1, p.second * temp / 100);
            prob.emplace_back(p.first, w);
            total += w;
        }
        logger.logDebug("概率计算完成, total=" + to_string(total) + ", prob.size=" + to_string(prob.size()));
        
        if (total <= 0) {
            logger.logDebug("total<=0 退出");
            break;
        }
        
        int r = rand() % total, sum = 0, sel = -1;
        for (auto& p : prob) {
            sum += p.second;
            if (sum > r) { sel = p.first; break; }
        }
        logger.logDebug("选择完成, sel=" + to_string(sel));
        
        if (sel == -1) {
            logger.logDebug("sel==-1 退出");
            break;
        }

        // ============================================================
        // 7. 预测误差计算与传播
        // ============================================================
        float probSelected = 0.0f;
        for (auto& p : prob) {
            if (p.first == sel) {
                probSelected = (float)p.second / total;
                break;
            }
        }
        probSelected = max(probSelected, 1e-6f);
        float surprise = -log(probSelected);
        surprise = max(0.0f, min(10.0f, surprise));
        
        PredictionRecord record;
        record.predictedToken = sel;
        record.actualToken = sel;
        record.predictionScore = probSelected;
        record.surprise = surprise;
        record.context = ctx;
        record.timestamp = chrono::system_clock::now();
        predictionHistory.push_back(record);
        if (predictionHistory.size() > PREDICTION_HISTORY_SIZE) {
            predictionHistory.pop_front();
        }

        if (!ctx.empty()) {
            logger.logDebug("调用 propagateTokenError, sel=" + to_string(sel) + ", surprise=" + to_string(surprise));
            propagateTokenError(sel, surprise, ctx, token);
            logger.logDebug("propagateTokenError 完成");
        }

        // ============================================================
        // 8. 记录边 + TD学习
        // ============================================================
        std::vector<DynamicEdge*> involvedEdges;
        if (!ctx.empty()) {
            int lastTid = ctx.back();
            Neuron* prevNeu = findNeuronByToken(lastTid);
            if (prevNeu) {
                DynamicEdge* e = prevNeu->findOutput(sel);
                if (e) involvedEdges.push_back(e);
            }
        }
        recordStepEdges(involvedEdges);

        int flu = calcFluency(ctx, token);
        stepFluency.push_back(flu);
		// ========== 新增：记录激活边 ==========
        // ========== 记录激活边（安全版） ==========
		if (!ctx.empty()) {
		    int lastTid = ctx.back();
		    Neuron* prevNeu = findNeuronByToken(lastTid);
		    if (prevNeu) {
		        int neuronId = prevNeu->neuronId;
		        
		        // 记录从 lastTid 到 sel 的边
		        DynamicEdge* edge = prevNeu->findOutput(sel);
		        if (edge) {
		            recordActivatedEdge(neuronId, sel);  // ← 存 (神经元ID, 目标token)
		        }
		        
		        // 记录所有有意义的边
		        for (auto& e : prevNeu->outputs) {
		            if (e.weight > 2) {
		                recordActivatedEdge(neuronId, e.target);
		            }
		        }
		    }
		}
		recordActivatedToken(sel);
        // ============================================================
        // 9. 更新上下文和记忆
        // ============================================================
        ctx.push_back(sel);
        addToShortMemory(sel);
        activateAmygdala(sel);
        logger.logDebug("上下文更新完成, ctx.size=" + to_string(ctx.size()));
        
        {
            lock_guard<recursive_mutex> lock(tokenMtx);
            if (sel >= 0 && sel < (int)token.vocabList.size() && token.vocabList[sel] == "。") {
                logger.logDebug("遇到句号，结束生成");
                break;
            }
        }

        if (sel >= 0 && sel < (int)token.vocabList.size()) {
            string word = token.vocabList[sel];
            if (!word.empty()) {
                auto posMap = token.posTagger.getPosMap(word);
                for (auto& pair : posMap) {
                    int tid = token.getTokenId(pair.first);
                    if (tid == sel) {
                        token.posCache[tid] = pair.second;
                        break;
                    }
                }
            }
        }

        // ============================================================
        // 10. 定期整合全局预测误差
        // ============================================================
        if (step > 0 && step % 10 == 0) {
            logger.logDebug("调用 integrateGlobalPredictionError");
            integrateGlobalPredictionError();
            logger.logDebug("integrateGlobalPredictionError 完成");
        }
        
        logger.logDebug("主循环第 " + to_string(step) + " 步完成");
    }

    logger.logDebug("主循环结束，共 " + to_string(step) + " 步");

    applyTDUpdates();
    logger.logDebug("applyTDUpdates 完成");
    
    curSubgraph.clear();
    logger.logDebug("curSubgraph.clear 完成");
    
    logger.logDebug("singleGenerate 结束，返回 ctx.size=" + to_string(ctx.size()));
    
    return ctx;
}
void BrainCortex::updateGains(const std::vector<int>& ctx, TextTokenizer& token) {
    logger.logDebug("updateGains 进入");
    
    float base = 1.0f;
    logger.logDebug("updateGains 计算 base");
    
    float err = getSelfValue("PredictionErrorGlobal") / 50.0f;
    logger.logDebug("updateGains 计算 err=" + to_string(err));
    
    err = std::max(0.0f, std::min(1.0f, err));
    logger.logDebug("updateGains err 裁剪后=" + to_string(err));

    alpha_L = base + err * 0.6f;
    logger.logDebug("updateGains alpha_L=" + to_string(alpha_L));
    
    if (curEmo.type == EMO_EXCITE) alpha_A = base + 0.4f;
    else alpha_A = base - err * 0.3f;
    logger.logDebug("updateGains alpha_A=" + to_string(alpha_A));
    
    alpha_R = base + (rand() % 20 - 10) / 50.0f;
    logger.logDebug("updateGains alpha_R=" + to_string(alpha_R));

    alpha_L = std::max(0.4f, std::min(1.6f, alpha_L));
    alpha_A = std::max(0.4f, std::min(1.6f, alpha_A));
    alpha_R = std::max(0.4f, std::min(1.6f, alpha_R));
    
    logger.logDebug("updateGains 完成");
}

// ===================== GrowingAGI =====================
class GrowingAGI {
public:
    TextTokenizer token;
    BrainCortex cortex;
    int dialogueCount=0;

    void initKnowledge();
    void train(const string& corpus);
    void appendExcellent(const string& text);
    vector<int> injectKnowledgeContext(const string& input);
    vector<int> singleGenerate(vector<int> ctx);
    vector<int> longTextGenerate(const string& prompt, int mode);
    vector<int> generate(const string& prompt, int mode);
    void saveModel(const string& path);
    bool loadModel(const string& path);
    float stringSimilarity(const string& a, const string& b);
    vector<int> getKnowledgeAnswer(const string& input);
    
    // ========== 新增：外部控制接口 ==========
    void setGoal(int goalToken) {
        cortex.setGoal(goalToken);
        if (goalToken >= 0 && goalToken < (int)token.vocabList.size()) {
            cout << "[系统] 设置目标为: " << token.vocabList[goalToken] << endl;
        } else {
            cout << "[系统] 设置目标 token: " << goalToken << endl;
        }
    }
    
    void setStyle(StyleIntent style) {
        cortex.setStyle(style);
        const char* styleName[] = {"STORY", "EMOTION", "SIMPLE"};
        cout << "[系统] 设置风格为: " << styleName[style] << endl;
    }
    
    string getConsciousReport() {
        return cortex.getConsciousReport();
    }
    
    void manualConsolidate(int seconds = 60) {
    	cortex.offlineConsolidation(token, seconds);
	}
    
    void onlineLearn(const string& text) {
        if (!text.empty()) {
            cout << "[系统] 开始在线学习..." << endl;
            auto seq = token.encode(text);
            cortex.onlineLearn(text, token);
            cout << "[系统] 在线学习完成！" << endl;
        }
    }
    
    // ========== 新增：状态查询 ==========
    void printStatus() {
        cout << "\n=== 系统状态 ===" << endl;
        cout << "神经元总数: " << cortex.getTotalNeurons() << endl;
        cout << "短时记忆大小: " << cortex.shortMemory.size() << endl;
        cout << "情绪状态: " << cortex.getEmotionStatus() << endl;
        cout << "当前风格: " << cortex.getStyleName() << endl;
        cout << "当前目标: ";
        if (cortex.getGoalIntent() >= 0 && cortex.getGoalIntent() < (int)token.vocabList.size()) {
            cout << token.vocabList[cortex.getGoalIntent()];
        } else {
            cout << "未设置";
        }
        cout << endl;
        cout << "激活水平: " << cortex.getSelfValue("AvgActivation") << endl;
        cout << "质量评分: " << cortex.getSelfValue("QualitySliding") << endl;
        cout << "\n" << getConsciousReport() << endl;
        cout << "================" << endl;
    }
};

// ===================== GrowingAGI 实现 =====================
void GrowingAGI::initKnowledge(){
    token.loadKnowledge();
    for(auto& kv:token.knowledgeVec){
        vector<int> ktok=token.encode(kv.first);
        int cid=token.createConcept(ktok,"knowledge");
        cortex.injectKnowledge(cid,kv.second,token);
    }
}/*
void GrowingAGI::initKnowledge(){
    // ========== 硬编码知识库（绕过文件） ==========
    vector<pair<string, string>> hardcodedKnowledge = {
        {"你好", "你好，很高兴认识你！"},
        {"天气", "今天天气不错，适合出门散步。"},
        {"再见", "再见，欢迎下次再来！"},
        // ... 你想加什么就加什么
    };
    
    // 直接注入到 token.knowledgeVec
    for (auto& entry : hardcodedKnowledge) {
        // 先确保词表里有这些字
        token.buildVocab(entry.first);
        token.buildVocab(entry.second);
        
        // 编码
        vector<int> ktok = token.encode(entry.first);
        vector<int> vtok = token.encode(entry.second);
        
        // 存到 knowledgeVec
        token.knowledgeVec.emplace_back(entry.first, vtok);
        
        // 创建概念
        int cid = token.createConcept(ktok, "knowledge");
        cortex.injectKnowledge(cid, vtok, token);
    }
}*/
void GrowingAGI::train(const string& corpus){
    cout<<"[1] 构建词表...\n"; token.buildVocab(corpus);
    cout<<"[2] 编码...\n"; auto seq=token.encode(corpus);
    cout<<"[3] 自学习（selfGrowth）...\n";
    for(int i=0; i<epochs; i++){
        cout << "  第 " << i+1 << " 轮学习..." << endl;
        cortex.selfGrowth(seq,token);
    }
    cout<<"[4] 加载知识库...\n"; initKnowledge();
    cout<<"[5] 提取短语和词性模板...\n";
    token.extractTemplatesFromCorpus(seq);
    token.extractTemplatesFromExcellentFile();
    cout<<"完成，神经元总数："<<cortex.totalNeurons()<<endl;
}

void GrowingAGI::appendExcellent(const string& text){
    if(text.empty() || (int)text.size()>200 || (int)text.size()<15) return;
    // 检查重复（读取文件时，文件内容为 UTF-8，需转为 GBK 比较）
    ifstream check(DYNAMIC_TRAIN_FILE);
    bool exist=false;
    if(check.is_open()){
        string line;
        while(getline(check, line)){
            string gbkLine = utf8ToGbk(line);  // 将 UTF-8 行转为 GBK 后再比较
            if(gbkLine == text){ exist = true; break; }
        }
        check.close();
    }
    if(exist) return;
    // 写入前将 GBK 转为 UTF-8
    string utf8Text = gbkToUtf8(text);
    ofstream f(DYNAMIC_TRAIN_FILE, ios::app);
    if(f){ f << utf8Text << "\n"; f.close(); }
    // 提取模板时使用原始 GBK text（内部处理）
    vector<int> seq = token.encode(text);
    token.extractTemplatesFromCorpus(seq);
}

vector<int> GrowingAGI::injectKnowledgeContext(const string& input){
    vector<int> add;
    vector<int> concepts=token.splitCompoundConcepts(input);
    for(int cid:concepts){ vector<int> content=token.getKnowledgeContent(cid); add.insert(add.end(),content.begin(),content.end()); }
    return add;
}

vector<int> GrowingAGI::singleGenerate(vector<int> ctx) {
    return cortex.singleGenerate(ctx, token);
}

vector<int> GrowingAGI::longTextGenerate(const string& prompt, int mode){
    vector<int> fullOutput, currentCtx=token.encode(prompt);
    vector<int> knowCtx=injectKnowledgeContext(prompt);
    currentCtx.insert(currentCtx.end(),knowCtx.begin(),knowCtx.end());
    cortex.saveLongTextSegment(currentCtx);
    vector<int> epRetrieved = cortex.retrieveEpisodes(currentCtx, token);
    if(!epRetrieved.empty()){
        currentCtx.insert(currentCtx.end(), epRetrieved.begin(), epRetrieved.end());
    }
    int sentenceCount=0;
    for(int p=0;p<3;p++){
        if(currentCtx.empty()) break;
        vector<int> inferred=cortex.runReasoning(currentCtx,hp.REASONING_DEPTH,token);
        currentCtx.insert(currentCtx.end(),inferred.begin(),inferred.end());
        vector<int> seg = singleGenerate(currentCtx);
        fullOutput.insert(fullOutput.end(),seg.begin(),seg.end());
        cortex.saveLongTextSegment(seg);
        cortex.storeEpisode(seg, token);
        sentenceCount++;
        if(sentenceCount>=3){
            for(int i=seg.size()-1;i>=0;i--){
                Neuron* n=cortex.findNeuronByToken(seg[i]);
                if(n && n->autoPos==POS_NOUN){ cortex.logicAnchorTokens.push_back(seg[i]); if(cortex.logicAnchorTokens.size()>5) cortex.logicAnchorTokens.pop_front(); break; }
            }
            sentenceCount=0;
        }
        currentCtx=seg;
        if(fullOutput.size()>hp.MAX_PARAGRAPH_LENGTH) break;
    }
    cortex.reflect(fullOutput,token);
    return fullOutput;
}

vector<int> GrowingAGI::generate(const string& prompt, int mode) {
    dialogueCount++;
    cortex.stepCounter++;
    // ============================================================
    // Step 0: 知识库模糊匹配
    // ============================================================
    vector<int> knowledgeAnswer = getKnowledgeAnswer(prompt);
    if (!knowledgeAnswer.empty()) {
        // 命中知识库，直接返回
        for (int tid : knowledgeAnswer) {
            cortex.addToShortMemory(tid);
        }
        cortex.storeEpisode(knowledgeAnswer, token);
        cortex.reflect(knowledgeAnswer, token);
        
        if (dialogueCount % hp.SAVE_EVERY_N_ROUNDS == 0) {
            saveModel(MODEL_FILE);
        }
        
        return knowledgeAnswer;
    }
    if (cortex.stepCounter % hp.INNER_GOAL_CHECK_INTERVAL == 0) {
        cortex.generateInnerGoals();
    }
    cortex.consciousBroadcast();

    vector<int> baseCtx = token.encode(prompt);
    vector<int> knowCtx = injectKnowledgeContext(prompt);
    baseCtx.insert(baseCtx.end(), knowCtx.begin(), knowCtx.end());

    cortex.curEmo.intensity = cortex.curEmo.intensity * 7 / 10;
    for (int& v : cortex.amygdalaActivation) v = v * 7 / 10;
    for (int tid : baseCtx) cortex.activateAmygdala(tid);

    if (cortex.goalIntent != -1) {
        vector<int> plan = cortex.planToGoal(baseCtx, cortex.goalIntent, hp.PLANNING_MAX_STEPS, token);
        if (!plan.empty() && plan.size() > baseCtx.size()) {
            baseCtx = plan;
        }
    }

    vector<int> best;
    int bestTotal = -1;
    for (int r = 0; r < hp.SELF_CHECK_ROUND; r++) {
        vector<int> res;
        if (prompt.size() > 10 || mode == 3)
            res = longTextGenerate(prompt, mode);
        else
            res = singleGenerate(baseCtx);

        int q = cortex.evaluateOutput(res, token, mode);
        int cnt = 0;
        for (int x : res) if (x > 3) cnt++;
        bool validLen = (cnt >= 6 && cnt <= 200);
        if (!validLen) continue;
        if (q < 25) continue;
        if (q > bestTotal) {
            bestTotal = q;
            best = res;
        }
    }

    if (best.empty()) {
        best = singleGenerate(baseCtx);
        if (bestTotal == -1) {
            bestTotal = cortex.evaluateOutput(best, token, mode);
        }
    }

    cortex.reflect(best, token);

    if (bestTotal != -1) {
        hp.recordQuality(bestTotal);
        hp.autoTune(bestTotal);
    }

    float inferenceConf = cortex.lastInferenceConfidence;
    static deque<int> topoChanges;
    bool hasTopoChange = (bestTotal < hp.NEUROGENESIS_QUALITY_THRESH);
    topoChanges.push_back(hasTopoChange ? 1 : 0);
    if (topoChanges.size() > 10) topoChanges.pop_front();
    float topoChangeRate = 0;
    for (int v : topoChanges) topoChangeRate += v;
    topoChangeRate /= topoChanges.size();
    cortex.updateSelfModel(bestTotal, inferenceConf, topoChangeRate);

    cortex.updateTwoLoopController(bestTotal);
    cortex.maybeMutateTopology(bestTotal);

    string txt = token.decode(best);
    int q = cortex.evaluateOutput(best, token);
    int f = cortex.calcFluency(best, token);
    if (q >= 55 && f >= 20 && (int)txt.size() > 15) {
        appendExcellent(txt);
    }
    if (dialogueCount % hp.SAVE_EVERY_N_ROUNDS == 0) {
        saveModel(MODEL_FILE);
    }

    return best;
}

void GrowingAGI::saveModel(const string& path){
    ofstream f(path,ios::binary); if(!f) return;
    // ========== 写入总神经元数 ==========
    int totalNeuronCount = cortex.totalNeurons();
    f.write((char*)&totalNeuronCount, sizeof(totalNeuronCount));
    { lock_guard<recursive_mutex> lock(tokenMtx); int vs=token.vocabList.size(); f.write((char*)&vs,sizeof(vs)); for(auto& s:token.vocabList){ int l=s.size(); f.write((char*)&l,sizeof(l)); f.write(s.data(),l); }
      int rs=token.radicalPool.size(); f.write((char*)&rs,sizeof(rs)); for(auto& s:token.radicalPool){ int l=s.size(); f.write((char*)&l,sizeof(l)); f.write(s.data(),l); }
      int cs=token.conceptSeq.size(); f.write((char*)&cs,sizeof(cs)); for(auto& vec:token.conceptSeq){ int t=vec.size(); f.write((char*)&t,sizeof(t)); for(int x:vec) f.write((char*)&x,sizeof(x)); }
      f.write((char*)&token.nextConceptId,sizeof(token.nextConceptId));
      int ptSize = token.phraseTemplates.size(); f.write((char*)&ptSize,sizeof(ptSize));
      for(auto& pt : token.phraseTemplates){
          int len = pt.first.size(); f.write((char*)&len,sizeof(len));
          for(int t : pt.first) f.write((char*)&t,sizeof(t));
          f.write((char*)&pt.second,sizeof(pt.second));
      }
      int skSize = token.posSkeletonTemplates.size(); f.write((char*)&skSize,sizeof(skSize));
      for(auto& sk : token.posSkeletonTemplates){
          f.write((char*)&sk.first,sizeof(sk.first));
          f.write((char*)&sk.second,sizeof(sk.second));
      }
    }
    { lock_guard<recursive_mutex> lock(cortexMtx); f.write((char*)&cortex.nextNeuronId,sizeof(cortex.nextNeuronId));
      for(auto& layer:cortex.layers){ int sz=layer.size(); f.write((char*)&sz,sizeof(sz)); for(auto& n:layer){
          f.write((char*)&n.neuronId,sizeof(n.neuronId)); f.write((char*)&n.layer,sizeof(n.layer));
          f.write((char*)&n.potential,sizeof(n.potential)); f.write((char*)&n.activation,sizeof(n.activation));
          f.write((char*)&n.inhibition,sizeof(n.inhibition)); f.write((char*)&n.energy,sizeof(n.energy));
          f.write((char*)&n.maturity,sizeof(n.maturity)); f.write((char*)&n.emotionEnergy,sizeof(n.emotionEnergy));
          f.write((char*)&n.autoPos,sizeof(n.autoPos)); f.write((const char*)n.posScore,sizeof(n.posScore));
          f.write((char*)&n.mode,sizeof(n.mode)); f.write((char*)&n.modeStrength,sizeof(n.modeStrength));
          int bt=n.boundTokens.size(); f.write((char*)&bt,sizeof(bt)); for(int x:n.boundTokens) f.write((char*)&x,sizeof(x));
          int pctSize = n.contextPositionPcts.size();
			f.write((char*)&pctSize, sizeof(pctSize));
			for (auto& kv : n.contextPositionPcts) {
			    int contextID = kv.first;
			    const vector<int>& positions = kv.second;
			    f.write((char*)&contextID, sizeof(contextID));
			    int posLen = positions.size();
			    f.write((char*)&posLen, sizeof(posLen));
			    for (int p : positions) {
			        f.write((char*)&p, sizeof(p));
			    }
			}
          int ts=n.tokenScoreVec.size(); f.write((char*)&ts,sizeof(ts)); for(auto& p:n.tokenScoreVec){ f.write((char*)&p.first,sizeof(p.first)); f.write((char*)&p.second,sizeof(p.second)); }
          int ins=n.inputs.size(); f.write((char*)&ins,sizeof(ins)); for(auto& e:n.inputs){ f.write((char*)&e.target,sizeof(e.target)); f.write((char*)&e.weight,sizeof(e.weight)); f.write((char*)&e.permanent,sizeof(e.permanent)); f.write((char*)&e.lifeCycle,sizeof(e.lifeCycle)); f.write((char*)&e.logic,sizeof(e.logic));f.write((char*)&e.decayAge,sizeof(e.decayAge)); }
          int outs=n.outputs.size(); f.write((char*)&outs,sizeof(outs)); for(auto& e:n.outputs){ f.write((char*)&e.target,sizeof(e.target)); f.write((char*)&e.weight,sizeof(e.weight)); f.write((char*)&e.permanent,sizeof(e.permanent)); f.write((char*)&e.lifeCycle,sizeof(e.lifeCycle)); f.write((char*)&e.logic,sizeof(e.logic));f.write((char*)&e.decayAge,sizeof(e.decayAge)); }
      } }
      int em = cortex.episodicMemory.size(); f.write((char*)&em,sizeof(em));
      for(auto& ep : cortex.episodicMemory){
          int ts = ep.tokens.size(); f.write((char*)&ts,sizeof(ts)); for(int t:ep.tokens) f.write((char*)&t,sizeof(t));
          f.write((char*)ep.embedding.data(), ep.embedding.size()*sizeof(float));
          float imp = ep.importance; f.write((char*)&imp,sizeof(imp));
      }
      int sn = cortex.selfNeurons.size(); f.write((char*)&sn,sizeof(sn));
      for(auto& snn : cortex.selfNeurons){
          int len = snn.name.size(); f.write((char*)&len,sizeof(len)); f.write(snn.name.data(), len);
          f.write((char*)&snn.value,sizeof(snn.value));
          f.write((char*)&snn.predicted,sizeof(snn.predicted));
          f.write((char*)&snn.error,sizeof(snn.error));
      }
      int qhSize = cortex.qualityHistoryForSelf.size(); f.write((char*)&qhSize,sizeof(qhSize));
      for(int qv : cortex.qualityHistoryForSelf) f.write((char*)&qv,sizeof(qv));
    }
    f.close();
}
// ========== 在 GrowingAGI 类里添加 ==========

// 1. 字符串相似度计算（Jaccard + 编辑距离混合）
float GrowingAGI::stringSimilarity(const string& a, const string& b) {
    if (a.empty() || b.empty()) return 0.0f;
    if (a == b) return 1.0f;
    
    // 如果一方完全包含另一方，给予高分
    if (a.find(b) != string::npos || b.find(a) != string::npos) {
        return 0.85f;
    }
    
    // Jaccard 相似度（字符级别）
    unordered_set<char> setA(a.begin(), a.end());
    unordered_set<char> setB(b.begin(), b.end());
    
    int intersection = 0;
    for (char c : setA) {
        if (setB.count(c)) intersection++;
    }
    int unionSize = setA.size() + setB.size() - intersection;
    float jaccard = unionSize == 0 ? 0.0f : (float)intersection / unionSize;
    
    // 如果 Jaccard 太低，直接返回
    if (jaccard < 0.2f) return jaccard;
    
    // 编辑距离（Levenshtein）
    int m = a.size(), n = b.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    int dist = dp[m][n];
    float maxLen = max(m, n);
    float normalizedDist = dist / maxLen;
    float levenshteinSim = 1.0f - normalizedDist;
    
    // 混合得分：Jaccard 权重 0.4，编辑距离权重 0.6
    return jaccard * 0.4f + levenshteinSim * 0.6f;
}

vector<int> GrowingAGI::getKnowledgeAnswer(const string& input) {
    const float MATCH_THRESHOLD = 0.35f;  // 匹配阈值
    
    // 先尝试精确匹配（快速路径）
    for (auto& kv : token.knowledgeVec) {
        if (input == kv.first) {
            cout << "[知识库] 精确命中: " << kv.first << endl;
            return kv.second;
        }
    }
    
    // 模糊匹配：遍历所有知识条目
    string bestKey;
    vector<int> bestValue;
    float bestScore = 0.0f;
    
    for (auto& kv : token.knowledgeVec) {
        const string& key = kv.first;
        
        // 计算相似度
        float sim = stringSimilarity(input, key);
        
        // 如果相似度超过阈值，记录最佳匹配
        if (sim >= MATCH_THRESHOLD && sim > bestScore) {
            bestScore = sim;
            bestKey = key;
            bestValue = kv.second;
        }
    }
    
    if (bestScore >= MATCH_THRESHOLD) {
        cout << "[知识库] 模糊命中: '" << input << "' → '" << bestKey 
             << "' (相似度: " << (int)(bestScore * 100) << "%)" << endl;
        return bestValue;
    }
    
    return {};  // 没找到
}
bool GrowingAGI::loadModel(const string& path) {
    logger.logImportant("========== 开始加载模型 ==========");
    logger.logDebug("模型路径: " + path);
    
    ifstream f(path, ios::binary);
    if (!f.is_open()) {
        logger.logError("无法打开模型文件: " + path);
        return false;
    }
    logger.logDebug("文件已打开");

    // 1. 检查文件大小
    f.seekg(0, ios::end);
    size_t fileSize = f.tellg();
    f.seekg(0, ios::beg);
    logger.logDebug("文件大小: " + to_string(fileSize) + " 字节 (" + to_string(fileSize / 1024.0) + " KB)");

    if (fileSize < 1024) {
        logger.logError("文件太小 (" + to_string(fileSize) + " 字节)，放弃加载");
        f.close();
        return false;
    }

    // 2. 读取总神经元数量（第一个 int）
    int totalNeuronCount;
    f.read((char*)&totalNeuronCount, sizeof(totalNeuronCount));
    logger.logImportant("总神经元数: " + to_string(totalNeuronCount));

    if (totalNeuronCount < 0 || totalNeuronCount > 10000000) {
        logger.logError("神经元数量异常: " + to_string(totalNeuronCount) + "，文件损坏");
        f.close();
        return false;
    }
    logger.logDebug("神经元数量检查通过 ?");

    // ========================================================================
    // 3. 读取 TextTokenizer 数据
    // ========================================================================
    logger.logDebug("开始读取 TextTokenizer 数据...");
    {
        lock_guard<recursive_mutex> lock(tokenMtx);
        token.vocabList.clear();
        token.radicalPool.clear();
        token.conceptSeq.clear();
        token.conceptContent.clear();

        // 3.1 vocabList
        int vs;
        f.read((char*)&vs, sizeof(vs));
        logger.logDebug("vocabList.size = " + to_string(vs));
        if (vs < 0 || vs > 200000) {
            logger.logError("vocabList 大小异常: " + to_string(vs));
            f.close();
            return false;
        }
        token.vocabList.reserve(vs);
        for (int i = 0; i < vs; i++) {
            int l;
            f.read((char*)&l, sizeof(l));
            if (l < 0 || l > 100) {
                logger.logError("vocabList[" + to_string(i) + "] 字符串长度异常: " + to_string(l));
                f.close();
                return false;
            }
            string s(l, 0);
            f.read(&s[0], l);
            token.vocabList.push_back(s);
        }
        logger.logDebug("vocabList 读取完成 ? (" + to_string(vs) + " 条)");

        // 3.2 radicalPool
        int rs;
        f.read((char*)&rs, sizeof(rs));
        logger.logDebug("radicalPool.size = " + to_string(rs));
        if (rs < 0 || rs > 100000) {
            logger.logError("radicalPool 大小异常: " + to_string(rs));
            f.close();
            return false;
        }
        for (int i = 0; i < rs; i++) {
            int l;
            f.read((char*)&l, sizeof(l));
            if (l < 0 || l > 100) {
                logger.logError("radicalPool[" + to_string(i) + "] 字符串长度异常: " + to_string(l));
                f.close();
                return false;
            }
            string s(l, 0);
            f.read(&s[0], l);
            token.radicalPool.insert(s);
        }
        logger.logDebug("radicalPool 读取完成 ? (" + to_string(rs) + " 条)");

        // 3.3 conceptSeq
        int cs;
        f.read((char*)&cs, sizeof(cs));
        logger.logDebug("conceptSeq.size = " + to_string(cs));
        if (cs < 0 || cs > 100000) {
            logger.logError("conceptSeq 大小异常: " + to_string(cs));
            f.close();
            return false;
        }
        token.conceptSeq.reserve(cs);
        token.conceptContent.reserve(cs);
        for (int i = 0; i < cs; i++) {
            int t;
            f.read((char*)&t, sizeof(t));
            if (t < 0 || t > 1000000) {
                logger.logError("conceptSeq[" + to_string(i) + "] 长度异常: " + to_string(t));
                f.close();
                return false;
            }
            vector<int> v(t);
            for (int j = 0; j < t; j++) {
                f.read((char*)&v[j], sizeof(v[j]));
            }
            token.conceptSeq.push_back(v);
            token.conceptContent.emplace_back();
        }
        logger.logDebug("conceptSeq 读取完成 ? (" + to_string(cs) + " 条)");

        // 3.4 nextConceptId
        f.read((char*)&token.nextConceptId, sizeof(token.nextConceptId));
        logger.logDebug("nextConceptId = " + to_string(token.nextConceptId));

        // 3.5 phraseTemplates
        int ptSize;
        f.read((char*)&ptSize, sizeof(ptSize));
        logger.logDebug("phraseTemplates.size = " + to_string(ptSize));
        if (ptSize < 0 || ptSize > 100000) {
            logger.logError("phraseTemplates 大小异常: " + to_string(ptSize));
            f.close();
            return false;
        }
        token.phraseTemplates.clear();
        token.phrasePrefix1.clear();
        token.phrasePrefix2.clear();
        for (int i = 0; i < ptSize; i++) {
            int len;
            f.read((char*)&len, sizeof(len));
            if (len < 0 || len > 10) {
                logger.logError("phraseTemplates[" + to_string(i) + "] 长度异常: " + to_string(len));
                f.close();
                return false;
            }
            vector<int> ph(len);
            for (int j = 0; j < len; j++) {
                f.read((char*)&ph[j], sizeof(ph[j]));
            }
            int cnt;
            f.read((char*)&cnt, sizeof(cnt));
            if (cnt < 0 || cnt > 10000) {
                logger.logError("phraseTemplates[" + to_string(i) + "] 计数异常: " + to_string(cnt));
                f.close();
                return false;
            }
            token.phraseTemplates.emplace_back(ph, cnt);
            if (!ph.empty()) {
                token.phrasePrefix1[ph[0]].push_back(i);
                if (ph.size() >= 2) {
                    token.phrasePrefix2[((uint64_t)ph[0] << 16) | ph[1]].push_back(i);
                }
            }
        }
        logger.logDebug("phraseTemplates 读取完成 ? (" + to_string(ptSize) + " 条)");

        // 3.6 posSkeletonTemplates
        int skSize;
        f.read((char*)&skSize, sizeof(skSize));
        logger.logDebug("posSkeletonTemplates.size = " + to_string(skSize));
        if (skSize < 0 || skSize > 100000) {
            logger.logError("posSkeletonTemplates 大小异常: " + to_string(skSize));
            f.close();
            return false;
        }
        token.posSkeletonTemplates.clear();
        for (int i = 0; i < skSize; i++) {
            uint64_t code;
            int cnt;
            f.read((char*)&code, sizeof(code));
            f.read((char*)&cnt, sizeof(cnt));
            if (cnt < 0 || cnt > 1000000) {
                logger.logError("posSkeletonTemplates[" + to_string(i) + "] 计数异常: " + to_string(cnt));
                f.close();
                return false;
            }
            token.posSkeletonTemplates.emplace_back(code, cnt);
        }
        logger.logDebug("posSkeletonTemplates 读取完成 ? (" + to_string(skSize) + " 条)");

        token.updateTokenCache();
        logger.logDebug("TextTokenizer 数据读取完成 ?");
    }

    // ========================================================================
    // 4. 读取 BrainCortex 数据
    // ========================================================================
    logger.logDebug("开始读取 BrainCortex 数据...");
    {
        lock_guard<recursive_mutex> lock(cortexMtx);
        for (auto& l : cortex.layers) l.clear();

        // 4.1 nextNeuronId
        f.read((char*)&cortex.nextNeuronId, sizeof(cortex.nextNeuronId));
        logger.logDebug("nextNeuronId = " + to_string(cortex.nextNeuronId));

        // 4.2 读取每一层
        for (int layerIdx = 0; layerIdx < (int)cortex.layers.size(); layerIdx++) {
            int sz;
            f.read((char*)&sz, sizeof(sz));
            logger.logDebug("层 " + to_string(layerIdx) + " 神经元数 = " + to_string(sz));

            if (sz < 0 || sz > 1000000) {
                logger.logError("层 " + to_string(layerIdx) + " 大小异常: " + to_string(sz));
                f.close();
                return false;
            }

            cortex.layers[layerIdx].reserve(sz);
            for (int i = 0; i < sz; i++) {
                Neuron n;

                // 读取基础字段
                f.read((char*)&n.neuronId, sizeof(n.neuronId));
                f.read((char*)&n.layer, sizeof(n.layer));
                f.read((char*)&n.potential, sizeof(n.potential));
                f.read((char*)&n.activation, sizeof(n.activation));
                f.read((char*)&n.inhibition, sizeof(n.inhibition));
                f.read((char*)&n.energy, sizeof(n.energy));
                f.read((char*)&n.maturity, sizeof(n.maturity));
                f.read((char*)&n.emotionEnergy, sizeof(n.emotionEnergy));
                f.read((char*)&n.autoPos, sizeof(n.autoPos));
                f.read((char*)n.posScore, sizeof(n.posScore));
                f.read((char*)&n.mode, sizeof(n.mode));
                f.read((char*)&n.modeStrength, sizeof(n.modeStrength));

                // 读取 boundTokens
                int bt;
                f.read((char*)&bt, sizeof(bt));
                for (int j = 0; j < bt; j++) {
                    int x;
                    f.read((char*)&x, sizeof(x));
                    n.boundTokens.insert(x);
                }

                // 读取 contextPositionPcts
                int pctSize;
                f.read((char*)&pctSize, sizeof(pctSize));
                for (int j = 0; j < pctSize; j++) {
                    int contextID;
                    f.read((char*)&contextID, sizeof(contextID));
                    int posLen;
                    f.read((char*)&posLen, sizeof(posLen));
                    vector<int> positions(posLen);
                    for (int k = 0; k < posLen; k++) {
                        f.read((char*)&positions[k], sizeof(positions[k]));
                    }
                    n.contextPositionPcts[contextID] = positions;
                }

                // 读取 tokenScoreVec
                int ts;
                f.read((char*)&ts, sizeof(ts));
                for (int j = 0; j < ts; j++) {
                    int k, v;
                    f.read((char*)&k, sizeof(k));
                    f.read((char*)&v, sizeof(v));
                    n.tokenScoreVec.emplace_back(k, v);
                }

                // 读取 inputs
                int ins;
                f.read((char*)&ins, sizeof(ins));
                if (ins < 0 || ins > 100000) {
                    logger.logError("神经元 " + to_string(n.neuronId) + " inputs 数量异常: " + to_string(ins));
                    f.close();
                    return false;
                }
                n.inputs.reserve(ins);
                for (int j = 0; j < ins; j++) {
                    DynamicEdge e;
                    f.read((char*)&e.target, sizeof(e.target));
                    f.read((char*)&e.weight, sizeof(e.weight));
                    f.read((char*)&e.permanent, sizeof(e.permanent));
                    f.read((char*)&e.lifeCycle, sizeof(e.lifeCycle));
                    f.read((char*)&e.logic, sizeof(e.logic));
                    f.read((char*)&e.decayAge, sizeof(e.decayAge)); 
                    n.inputs.push_back(e);
                }

                // 读取 outputs
                int outs;
                f.read((char*)&outs, sizeof(outs));
                if (outs < 0 || outs > 100000) {
                    logger.logError("神经元 " + to_string(n.neuronId) + " outputs 数量异常: " + to_string(outs));
                    f.close();
                    return false;
                }
                n.outputs.reserve(outs);
                for (int j = 0; j < outs; j++) {
                    DynamicEdge e;
                    f.read((char*)&e.target, sizeof(e.target));
                    f.read((char*)&e.weight, sizeof(e.weight));
                    f.read((char*)&e.permanent, sizeof(e.permanent));
                    f.read((char*)&e.lifeCycle, sizeof(e.lifeCycle));
                    f.read((char*)&e.logic, sizeof(e.logic));
                    f.read((char*)&e.decayAge, sizeof(e.decayAge)); 
                    n.outputs.push_back(e);
                }

                cortex.layers[layerIdx].push_back(n);
            }
            logger.logDebug("层 " + to_string(layerIdx) + " 读取完成 ? (" + to_string(sz) + " 个神经元)");
        }

        // 4.3 episodicMemory
        int em;
        f.read((char*)&em, sizeof(em));
        logger.logDebug("episodicMemory.size = " + to_string(em));
        cortex.episodicMemory.clear();
        for (int i = 0; i < em; i++) {
            int ts;
            f.read((char*)&ts, sizeof(ts));
            vector<int> tokens(ts);
            for (int j = 0; j < ts; j++) {
                f.read((char*)&tokens[j], sizeof(tokens[j]));
            }
            vector<float> emb(token.EMBED_DIM);
            f.read((char*)emb.data(), emb.size() * sizeof(float));
            float imp;
            f.read((char*)&imp, sizeof(imp));
            cortex.episodicMemory.push_back({tokens, emb, chrono::system_clock::now(), imp});
        }
        logger.logDebug("episodicMemory 读取完成 ? (" + to_string(em) + " 条)");

        // 4.4 selfNeurons
        int sn;
        f.read((char*)&sn, sizeof(sn));
        logger.logDebug("selfNeurons.size = " + to_string(sn));
        cortex.selfNeurons.clear();
        for (int i = 0; i < sn; i++) {
            int len;
            f.read((char*)&len, sizeof(len));
            string name(len, 0);
            f.read(&name[0], len);
            BrainCortex::SelfNeuron snn;
            snn.name = name;
            f.read((char*)&snn.value, sizeof(snn.value));
            f.read((char*)&snn.predicted, sizeof(snn.predicted));
            f.read((char*)&snn.error, sizeof(snn.error));
            cortex.selfNeurons.push_back(snn);
        }
        logger.logDebug("selfNeurons 读取完成 ? (" + to_string(sn) + " 条)");

        // 4.5 qualityHistoryForSelf
        int qhSize;
        f.read((char*)&qhSize, sizeof(qhSize));
        logger.logDebug("qualityHistoryForSelf.size = " + to_string(qhSize));
        cortex.qualityHistoryForSelf.clear();
        for (int i = 0; i < qhSize; i++) {
            int qv;
            f.read((char*)&qv, sizeof(qv));
            cortex.qualityHistoryForSelf.push_back(qv);
        }
        logger.logDebug("qualityHistoryForSelf 读取完成 ? (" + to_string(qhSize) + " 条)");

        // 补齐 selfNeurons
        while ((int)cortex.selfNeurons.size() < hp.SELF_NEURON_COUNT) {
            BrainCortex::SelfNeuron extra{"Extra", 0, 0, 0};
            cortex.selfNeurons.push_back(extra);
        }

        cortex.rebuildLayerIndex();
        logger.logDebug("BrainCortex 数据读取完成 ?");
    }

    f.close();
    logger.logImportant("========== 模型加载成功！?? ==========");
    return true;
}

string readFile(const string& filePath){
    ifstream f(filePath, ios::binary);
    if(!f){ cerr<<"错误：找不到 train.txt 文件！"<<endl; return ""; }
    stringstream ss; ss<<f.rdbuf();
    string utf8Content = ss.str();
    // UTF-8 -> GBK（内部使用）
    string gbkContent = utf8ToGbk(utf8Content);
    return gbkContent;
}

// ===================== 主函数 =====================
void showHelp() {
    cout << "\n=== 命令列表 ===" << endl;
    cout << "exit/quit     - 退出程序" << endl;
    cout << "/goal         - 设置生成目标 (例如: /goal 爱)" << endl;
    cout << "/style        - 设置生成风格 (0=STORY, 1=EMOTION, 2=SIMPLE)" << endl;
    cout << "/status       - 显示系统状态" << endl;
    cout << "/consolidate  - 手动离线巩固 (持续2分钟)" << endl;
    cout << "/train        - 在线学习新文本" << endl;
    cout << "/clear        - 清空短时记忆" << endl;
    cout << "/help         - 显示此帮助" << endl;
    cout << "直接输入文本  - 生成回复" << endl;
    cout << "================" << endl;
}

int main(){
	system("chcp 936"); 
	SetEnvironmentVariableA("OMP_STACKSIZE", "16M");
    srand((unsigned int)time(NULL));
    GrowingAGI agi;

    // ========== 询问是否启用日志 ==========
    cout << "===== 欢迎使用 AGI 系统 =====" << endl;
    cout << "是否启用详细日志？（将写入 log.txt）" << endl;
    cout << "输入 是/否 (y/n): ";
    
    string enlog;
    cin >> enlog;
    cin.ignore();  // 清除换行符
    
    // 判断用户输入
    if (enlog == "是" || enlog == "yes" || enlog == "y" || enlog == "Y" || enlog == "Yes") {
        logger.enable("log.txt");
    } else {
        logger.disable();
        cout << "日志已关闭，将只显示关键信息。" << endl;
    }
    cout << "是否启用日志输出？（将把所有信息打印至控制台）" << endl;
    cout << "输入 是/否 (y/n): ";
    string enlogoutput;
    cin >> enlogoutput;
    cin.ignore();  // 清除换行符
    if (enlogoutput == "是" || enlogoutput == "yes" || enlogoutput == "y" || enlogoutput == "Y" || enlogoutput == "Yes") {
	    logger.enableConsole(true);
	    cout << "控制台输出已开启，将显示所有调试信息。" << endl;
	} else {
	    logger.enableConsole(false);
	    cout << "控制台输出已关闭，只显示关键信息。" << endl;
	}
    cout << "=============================" << endl;

    // 显示工作目录
    char buffer[256];
    GetCurrentDirectoryA(256, buffer);
    cout << "当前工作目录: " << buffer << endl;

    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    cout << "可用物理内存: " << memInfo.ullAvailPhys / (1024*1024) << " MB" << endl;
	cout << "==== 测试知识库 ====" << endl;
    cout << "知识库文件: " << KNOWLEDGE_FILE << endl;
    // 加载或训练模型
    void* reserveMemory = malloc(100 * 1024 * 1024); 
    memset(reserveMemory, 0, 100 * 1024 * 1024); 
    if (agi.loadModel(MODEL_FILE)) {
        cout << "===== 模型加载成功 =====" << endl;
    } else {
        cout << "===== 从 train.txt 加载训练数据 =====" << endl;
        string corpus = readFile("train.txt");
        if (corpus.empty()) {
            cout << "训练数据为空，程序退出！" << endl;
            return -1;
        }
        agi.train(corpus);
        cout << "===== 训练完成 =====" << endl;
    }
	free(reserveMemory);
    showHelp();

    auto lastInputTime = chrono::steady_clock::now();
    bool inSleep = false;

    while (true) {
        cout << "\n请输入：";
        string input;
        if (cin.peek() != EOF) {
            getline(cin, input);
            lastInputTime = chrono::steady_clock::now();
            userInputWaiting = false;

            if (input == "exit" || input == "quit") break;
            
            // ========== 新增命令：手动开关日志 ==========
            if (input == "/log on") {
                logger.enable("log.txt");
                continue;
            }
            else if (input == "/log off") {
                logger.disable();
                continue;
            }
            else if (input == "/log status") {
			    cout << "文件日志: " << (logger.isFileEnabled() ? "已启用" : "已禁用") << endl;
			    cout << "控制台输出: " << (logger.isConsoleEnabled() ? "已开启" : "已关闭") << endl;
			    continue;
			}
            else if (input == "/help") {
                showHelp();
                continue;
            }
            else if (input == "/goal") {
                cout << "请输入目标文本: ";
                string goalStr;
                getline(cin, goalStr);
                if (!goalStr.empty()) {
                    auto seq = agi.token.encode(goalStr);
                    if (!seq.empty()) {
                        int goalToken = seq.back();
                        agi.setGoal(goalToken);
                    } else {
                        cout << "[错误] 无法编码目标文本" << endl;
                    }
                }
                continue;
            }
            else if (input == "/style") {
                cout << "选择风格 (0=STORY, 1=EMOTION, 2=SIMPLE): ";
                int style;
                cin >> style;
                cin.ignore();
                if (style >= 0 && style <= 2) {
                    agi.setStyle((StyleIntent)style);
                } else {
                    cout << "[错误] 无效的风格选择" << endl;
                }
                continue;
            }
            else if (input == "/status") {
                agi.printStatus();
                continue;
            }
            else if (input == "/consolidate") {
                cout << "开始离线巩固... (将持续2分钟，输入文本可中断)" << endl;
                userInputWaiting = true;
                thread t([&agi]() {
                    agi.manualConsolidate(120);
                    userInputWaiting = false;
                });
                t.detach();
                continue;
            }
            else if (input == "/train") {
                cout << "请输入训练文本: ";
                string trainText; 
                getline(cin, trainText);
                if (!trainText.empty()) {
                    agi.onlineLearn(trainText);
                }
                continue;
            }
            else if (input == "/clear") {
                agi.cortex.shortMemory.clear();
                agi.cortex.logicAnchorTokens.clear();
                agi.cortex.workspace.clear();
                cout << "[系统] 短时记忆已清空" << endl;
                continue;
            }
            else if (input.empty()) {
                continue;
            }

            // ========== 正常生成 ==========
            if (inSleep) {
                inSleep = false;
                cout << "[系统] 已从休眠中唤醒。" << endl;
            }

            agi.token.analyzeText(input);

            auto outputIds = agi.generate(input, 3);
            string output = agi.token.decode(outputIds);
            cout << "输出：" << output << endl;

			// ========== 新增：反馈提示 ==========
			cout << "评价此输出？(+ 满意 / - 不满意 / 回车跳过): ";
			string feedback;
			getline(cin, feedback);
			
			// 在 main() 或 GrowingAGI 的反馈处理中
			if (feedback == "+") {
			    agi.cortex.reinforceActivatedEdges(5);
			    userModel.recordTurn(agi.token.encode(input), outputIds, true);
			    agi.saveModel(MODEL_FILE);
			} else if (feedback == "-") {
			    agi.cortex.punishActivatedEdges(5);
			    userModel.recordTurn(agi.token.encode(input), outputIds, false);
			    agi.saveModel(MODEL_FILE);
			}
			
            agi.token.analyzeText(output);

        } else {
            auto idle = chrono::duration_cast<chrono::seconds>(
                chrono::steady_clock::now() - lastInputTime
            ).count();
            if (!inSleep && idle >= 60) {
                inSleep = true;
                userInputWaiting = false;
                thread consolidate([&agi]() {
                    agi.manualConsolidate(300);
                });
                consolidate.detach();
            }
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

    cout << "\n正在保存模型..." << endl;
    agi.saveModel(MODEL_FILE);
    cout << "模型已保存，再见！" << endl;
    return 0;
}
