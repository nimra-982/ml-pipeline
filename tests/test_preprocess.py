from src.preprocess import load_data, preprocess_data

def test_load_data():
    df = load_data()
    assert df.shape[0] > 0

def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0

