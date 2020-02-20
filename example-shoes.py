from jenga.tasks.shoes import ShoeCategorizationTask
from jenga.corruptions.image import GaussianNoise

task = ShoeCategorizationTask()

train_data = task.train_images
y_train = task.train_labels

model = task.fit_baseline_model(train_data, y_train)

predicted_classes = model.predict_classes(task.test_images)
print("Accuracy on clean test set", task.score_on_test_images(predicted_classes))

corruption = GaussianNoise(fraction=0.8)
noisy_test_images = corruption.transform(task.test_images)

predicted_classes = model.predict_classes(noisy_test_images)
print("Accuracy on noisy test set", task.score_on_test_images(predicted_classes))