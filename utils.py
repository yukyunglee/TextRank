import pickle


def read_dataset(file_directory):

    with open(file_directory, "rb") as f:
        dataset = pickle.load(f)

    return dataset


def save_result(save_directory, file):

    with open(save_directory, "wb") as f:
        pickle.dump(file, f)
