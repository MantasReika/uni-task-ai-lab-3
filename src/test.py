import os

from imageai.Classification.Custom import CustomImageClassification

execution_path = os.getcwd()
print("cwd")
print(execution_path)
prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("dataset/models/model_ex-073_acc-1.000000.h5")
prediction.setJsonPath("dataset/json/model_class.json")
prediction.loadModel(num_objects=3)

predictions, probabilities = prediction.predictImage("dataset/train/orange/orange_2.jpg", result_count=1)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
