data <- read.csv("simudata.csv")
library(rpart)
library(rpart.plot)
library(ggplot2)
library(caret)      # 用于数据划分和混淆矩阵
library(pROC)



# 查看数据结构和摘要
print("数据结构 (str):")
str(data)
print("数据摘要 (summary):")
summary(data)


# ---------------------------------------------------------------------------
# 任务 3：信用数据分析（R 语言实现）
# ---------------------------------------------------------------------------

# 0. 加载所需的库


# ---------------------------------------------------------------------------
# (3.1) 读入数据并了解各个自变量的含义
# ---------------------------------------------------------------------------

# (您已经完成了这一步)
# data <- read.csv("simudata.csv")
# 
# 为了确保代码可运行，我们假设 'data' 对象已存在于您的环境中。
# 我们需要将 'black' 转换为因子 (factor) 以便绘图和建模。

data$black_factor <- as.factor(data$black)

# ---------------------------------------------------------------------------
# (3.2) 对比箱线图并分析解读
# ---------------------------------------------------------------------------
# 变量已确定:
# "借贷比例" = creded
# "所有行为最大值" = maxpay

# 绘制 'creded' (借贷比例) 的对比箱线图
p1 <- ggplot(data, aes(x = black_factor, y = creded, fill = black_factor)) +
  geom_boxplot() +
  labs(title = "Boxplot of 'creded' (借贷比例) by Black",
       x = "Black (0=非违约, 1=违约)",
       y = "creded") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") # 添加一些颜色

print(p1)

# 绘制 'maxpay' (所有行为最大值) 的对比箱线图
p2 <- ggplot(data, aes(x = black_factor, y = maxpay, fill = black_factor)) +
  geom_boxplot() +
  labs(title = "Boxplot of 'maxpay' (所有行为最大值) by Black",
       x = "Black (0=非违约, 1=违约)",
       y = "maxpay") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  # 由于 'maxpay' 的值可能非常大且分布不均，对y轴进行log转换可能看得更清楚
  # 如果log转换后报错 (因为有0或负值)，请注释掉下面这行
  scale_y_log10() 

print(p2)

# ---
# (3.2) 解读 (示例):
# 
# 1.  **解读 'creded' (借贷比例):**
#     * 请观察 p1 图。
#     * 比较两组（"0" 和 "1"）的箱体（中位数、IQR）。
#     * 如果违约组 ("1") 的 'creded' 中位数显著高于非违约组 ("0")，这可能意味着：
#         > “借贷比例越高的用户，违约的风险也越高。”
# 
# 2.  **解读 'maxpay' (所有行为最大值):**
#     * 请观察 p2 图 (可能是 log 转换后的图)。
#     * 比较两组的中位数和分布。
#     * 如果两组差异不大，可能意味着：
#         > “单次最大支付金额与是否违约没有明显关系。”
#     * 如果违约组 ("1") 的 'maxpay' 显著更高或更低，请描述该趋势。
# ---

# ---------------------------------------------------------------------------
# (3.3) 划分训练集和测试集，建模型，计算 AUC 和混淆矩阵
# ---------------------------------------------------------------------------

# 设定随机种子 = 2024
set.seed(2024)

# 按 7:3 比例划分 (使用 black_factor 进行分层抽样)
trainIndex <- createDataPartition(data$black_factor, 
                                  p = .7, 
                                  list = FALSE, 
                                  times = 1)

trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

print(paste("Training data rows:", nrow(trainData)))
print(paste("Testing data rows:", nrow(testData)))

# 建立决策树模型 (使用 rpart 函数，设置默认参数)
# 我们使用 'black_factor' 作为因变量，'~ .' 使用所有其他变量
# rpart 会自动忽略 'black' (原始int变量)，因为它与 'black_factor' 相关
tree_model <- rpart(black_factor ~ ., 
                    data = trainData, 
                    method = "class")

# --- 在测试集上进行测试 ---

# 1. 预测概率 (用于 ROC 和 AUC)
#    type = "prob" 返回矩阵，第二列是 '1' (违约) 的概率
pred_prob <- predict(tree_model, testData, type = "prob")[, 2]

# 2. 预测类别 (用于混淆矩阵)
pred_class <- predict(tree_model, testData, type = "class")

# --- 计算 ROC 曲线和 AUC 值 ---
roc_curve <- roc(testData$black_factor, pred_prob)

print("AUC (Area Under Curve):")
auc_value <- auc(roc_curve)
print(auc_value)

# 绘制 ROC 曲线
plot(roc_curve, main = "ROC Curve", print.auc = TRUE, col = "blue")

# --- 计算并解释混淆矩阵 ---
# 确保指定 '1' (违约) 为 "Positive" class
cm <- confusionMatrix(data = pred_class, 
                      reference = testData$black_factor,
                      positive = "1") 

print("Confusion Matrix:")
print(cm)

# ---
# (3.3) 混淆矩阵解读:
# 
# 假设混淆矩阵如下:
#           Reference
# Prediction   0   1
#          0  TN  FN  (TN=真阴性, FN=假阴性)
#          1  FP  TP  (FP=假阳性, TP=真阳性)
#
# * **Accuracy (准确率):** (TP+TN) / (总数)。
#     * 解读: “模型在所有样本中预测正确的比例。”
# * **Sensitivity (召回率/真正率):** TP / (TP+FN) (在 cm$byClass 中查看)。
#     * 解读: “在所有**真实违约**的用户中，模型成功识别出了 [Sensitivity] %。”
# * **Specificity (特异度/真负率):** TN / (TN+FP) (在 cm$byClass 中查看)。
#     * 解读: “在所有**真实未违约**的用户中，模型成功识别出了 [Specificity] %。”
# * **Pos Pred Value (Precision/精确率):** TP / (TP+FP) (在 cm$byClass 中查看)。
#     * 解读: “在所有被模型**预测为违约**的用户中，有 [Precision] % 是真的违约了。”
# ---

# ---------------------------------------------------------------------------
# (3.4) 画出决策树的图形，并进行解读
# ---------------------------------------------------------------------------

# 使用 rpart.plot 绘制决策树
print("Generating Decision Tree Plot...")
rpart.plot(tree_model, 
           main = "Decision Tree for Credit Risk", 
           type = 3,  
           extra = 101, # 显示类别比例和百分比
           box.palette = "BuGn",
           shadow.col = "gray",
           yesno = 2)   # 显示 'yes'/'no'

# ---
# (3.4) 决策树解读:
# 
# 1.  **根节点 (最顶端的节点):**
#     * 它显示了整个训练集的平均违约率 (例如 "1" 占 33%)。
#
# 2.  **第一个分裂 (最重要的变量):**
#     * 查看根节点下面的第一个分裂条件 (例如 `creded < 0.15`)。
#     * 这是模型找到的最重要的单一预测变量。
#
# 3.  **决策路径:**
#     * 从根节点沿着 'yes'/'no' 路径到达一个叶节点，这条路径就是一条分类规则。
#     * 例如: "如果 `creded < 0.15` (yes) 且 `maxpay < 100000` (yes)，则预测为 0 (非违约)"。
#
# 4.  **叶节点 (底部的方框):**
#     * 显示该规则下的最终预测 (如 "0" 或 "1")。
#     * 显示该节点的样本纯度 (例如 "1 (80%)"，意味着落入此节点的 80% 样本是真的 "1")。
#     * 显示该节点占总样本的百分比 (例如 "25%"，意味着 25% 的训练数据遵循这条规则)。
#
# 5.  **变量重要性 (总结):**
#     * 在树的上层出现的变量 (如 `creded`, `maxpay`) 通常比较低层级的变量更重要。
# ---

warnings()