from preprocess.file_extraction import create_dataset_list, split_train_val_test

mapped_images_folder = "data"
dataset_list = create_dataset_list(mapped_images_folder)
train_conversation_dataset, validation_conversation_dataset, test_conversation_dataset = split_train_val_test(dataset_list, 
                                                                                                              train_json_path="train_cache/train_conversation.json", 
                                                                                                              validation_json_path="train_cache/validation_conversation.json", 
                                                                                                              test_json_path="train_cache/test_conversation.json")