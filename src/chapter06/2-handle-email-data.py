# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/4/4 12:35
# @Author     : white
# @FileName   : 2-handle-email-data.py
# @Software   : PyCharm
# **************************************
import pandas as pd

data_file_path = "sms_spam_collection\SMSSpamCollection.tsv"

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df["Label"].value_counts())


# Listing 6.2 Creating a balanced dataset
def create_balanced_dataset(df):
    """
    进行随机下采样，将两个数据集采样为一样大小
    :param df:
    :return:
    """
    num_spam = df[df["Label"] == "spam"].shape[0]  # 统计垃圾短信的实例数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)  # 随机抽取正常邮件实例，使其数量与垃圾短信实例相同。
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])  # 将正常短信子集与垃圾短信合并
    return balanced_df


balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# 进行标签映射
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})


# Listing 6.3 Splitting the dataset
def random_split(df, train_frac, validation_frac):
    """
    进行数据切割
    :param df:
    :param train_frac:
    :param validation_frac:
    :return:
    """
    # 数据打乱
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # 将整个 DataFrame 随机打乱

    train_end = int(len(df) * train_frac)  # 计算数据分割的索引
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]  # 分割 DataFrame
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 测试集默认大小为 0.2（即剩余部分）
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
