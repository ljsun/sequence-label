## Sequence label

数据来源于[李宏毅 2017 fall course ](https://docs.google.com/presentation/d/1k5-gTusho2SQdtcB2p3fiBwNiYVqou00p1QOi9uO_-4/edit#slide=id.g276b4d886c_0_0)，具体的数据说明参照该链接中的slides。

---

### new_data

处理后的数据存放在new_data目录，但是由于数据量超过github限制，因此无法上传。读者可自行处理数据。

原始数据与处理后的数据相比

- 将train.lab文件中的48 phone 替换为39 phone
- 将train.ark与train.lab每一行都相对应

### network structure

LSTM + DNN + Softmax

### problem

- 由于每一个sample的time_step长度不同，因此选择使用dynamic_rnn，但是并不知道dynamic_rnn在进行BP时，是否能够记得batch中每一个sample的time_step。关于这一点，正在看文档中。

### have done

- 使用pandas对数据预处理，处理后的数据保存在new_data目录中，原始数据在data目录中
- 实现对数据的batch读取
- 只使用train data进行训练（仅仅跑通网络），并未用到valid data和test data

---

### will do

- 由于没有test label，决定从原始的train data分出valid data和test data

---

### how to run

> enter main directory，then execute python train_rnn.py



