import sys
import os
import pandas as pd
from src.exception import customException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:    
            base_dir = os.path.dirname(os.path.abspath(__file__))

            project_root = os.path.abspath(
            os.path.join(base_dir, "..", "..")
            )

            artifacts_dir = os.path.join(project_root, "artifacts")

            model_path = os.path.join(artifacts_dir, "model.pkl")
            preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise customException(e,sys)

class customData:
    def __init__(
            self,
            gender:str,
            race_ethnicity:int,
            parental_level_of_education,
            lunch:str,
            test_preparation_course:str,
            reading_score:int,
            writing_score:int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise customException(e, sys)

        