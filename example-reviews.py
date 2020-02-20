from jenga.tasks.reviews import VideogameReviewsTask
from jenga.corruptions.text import MissingValues, BrokenCharacters

task = VideogameReviewsTask()

# This task has data for four weeks,
# We train on past reviews and try to predict the five star reviews for the next week
while task.advance_current_week():

    print("----- Week", task.current_week(), "-----")

    train_data = task.current_accumulated_train_data()
    train_labels = task.current_accumulated_train_labels()

    model = task.fit_baseline_model(train_data, train_labels)

    test_data = task.current_test_data()
    predictions = model.predict_proba(task.current_test_data())

    print("\tAUC on test data", task.score_on_current_test_data(predictions))

    corruption = MissingValues(column='title_and_review_text', fraction=0.8, na_value='')

    corrupted_test_data = corruption.transform(test_data)
    predictions = model.predict_proba(corrupted_test_data)
    print("\tAUC on corrupted test data (missing values)", task.score_on_current_test_data(predictions))

    corruption = BrokenCharacters(column='title_and_review_text', fraction=0.8)

    corrupted_test_data = corruption.transform(test_data)
    predictions = model.predict_proba(corrupted_test_data)
    print("\tAUC on corrupted test data (characters)", task.score_on_current_test_data(predictions))
