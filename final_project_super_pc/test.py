import tensorflow as tf

# 定义每个特征的描述
feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string),  # ID 是字符串
    'primary': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),  # 主序列是可变长度的整数
    'tertiary': tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),  # 三维坐标是 [x, y, z] 的浮点数组
}

# 解析函数
def _parse_function(example_proto):
    # 使用 feature_description 来解析数据
    return tf.io.parse_single_example(example_proto, feature_description)

# 读取 TFRecord 文件
tfrecord_file = "training.tfr"
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# 解析每个样本
parsed_dataset = raw_dataset.map(_parse_function)

# 迭代读取解析后的数据
for parsed_record in parsed_dataset:
    print(parsed_record)
