from utils.bedrock import BedrockLLMWrapper
from utils.database import DatabaseUtil
from utils.util import Util
import json
import time
import pandas as pd

class EvaluationTaskRunner:
    def __init__(self, eval_df: pd.DataFrame, 
                 model_id:str = 'anthropic.claude-3-sonnet-20240229-v1:0',
                 eval_model_id:str = 'anthropic.claude-3-sonnet-20240229-v1:0',
                 temperature: float = 0.0,
                 max_token_count: int = 2000,
                 max_attempts: int = 3, 
                 prompt_template_1: str = '',
                 prompt_template_2: str = '',
                 prompt_template_3: str = '',
                 few_shot_examples: str = '',
                 prompt_eval_system: str = '',
                 prompt_eval_template: str = '',
                 region: str = 'us-east-1',
                 csv_file_path: str = ''
                 ):
        self.eval_df = eval_df
        self.model_id = model_id
        self.eval_model_id = eval_model_id
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.prompt_template_1 = prompt_template_1
        self.prompt_template_2 = prompt_template_2
        self.prompt_template_3 = prompt_template_3
        self.few_shot_examples = few_shot_examples
        self.prompt_eval_system = prompt_eval_system
        self.prompt_eval_template = prompt_eval_template
        self.region = region
        self.csv_file_path = csv_file_path
        self.llm = BedrockLLMWrapper(model_id=self.model_id, 
                                             max_token_count=self.max_token_count,
                                             temperature=self.temperature,
                                             region=self.region
                                             )
        
        
        self.eval_llm = BedrockLLMWrapper(model_id=self.eval_model_id,
                                          system_prompt=self.prompt_eval_system,
                                          max_token_count=self.max_token_count,
                                          temperature=self.temperature,
                                          region=self.region
                                             )
        self.util = Util()


    def get_prompt(self, table_schema):
        if self.few_shot_examples != '':
            prompt = self.prompt_template_1.format(
                    TABLE_SCHEMA=table_schema,
                    FEW_SHOT_EXAMPLES=self.few_shot_examples,
                ) 
        else:
            prompt = self.prompt_template_1.format(
                    TABLE_SCHEMA=table_schema,
                ) 
        return prompt


    def build_grader_prompt(self, 
                            ground_truth_anomalies: str, 
                            table_schema: str, 
                            anomaly_detection_response:str 
                            ):
    
        prompt = self.prompt_eval_template.format(
                    ground_truth_anomalies=ground_truth_anomalies,
                    table_schema= table_schema,
                    anomaly_detection_response= anomaly_detection_response,
                ) 
        return prompt


    def run(self) -> pd.DataFrame:
        df = pd.DataFrame(self.eval_df)
        results = []
        
        # Prepare prompts for all questions
        prompts = []
        attachments = []
        for _, row in df.iterrows():
            inputfile: str = row['inputfile']
            table_schema: str = row['table_schema']
            # num_anomalies: str = row['num_anomalies']
            # anomalies: str = row['anomalies']

            model_prompt = self.get_prompt(table_schema=table_schema)
            prompts.append(model_prompt)
            # print(f'model_prompt: {model_prompt}')
            attachments.append(inputfile)
            # print(f'attachment: {inputfile}')

        # Generate SQL queries using threaded approach
        anomaly_detection_responses_1 = self.llm.generate_threaded(prompts=prompts,attachments=attachments,max_workers=5)
        
        # run the second prompt iterate through anomaly_detection_responses
        if self.prompt_template_2 != '':
            prompts_2 = []
            for anomaly_detection_response in anomaly_detection_responses_1:
                prompt_2 = self.prompt_template_2.format(PRIOR_RESPONSE=anomaly_detection_response[0])
                prompts_2.append(prompt_2)
            anomaly_detection_responses_2 = self.llm.generate_threaded(prompts=prompts_2,attachments=attachments,max_workers=5)

            if self.prompt_template_3 != '':
                prompts_3 = []
                for anomaly_detection_response in anomaly_detection_responses_2:
                    prompt_3 = self.prompt_template_3.format(PRIOR_RESPONSE=anomaly_detection_response[0])
                    prompts_3.append(prompt_3)

                anomaly_detection_responses_3 = self.llm.generate_threaded(prompts=prompts_3,attachments=attachments,max_workers=5)

        else:
            # create anomaly_detection_responses_2 and anomaly_detection_responses_3 with equal length    
            anomaly_detection_responses_2 = anomaly_detection_responses_1
            anomaly_detection_responses_3 = anomaly_detection_responses_1

        # bottleneck: from here on we are back to processing in sequence
        for index, (anomaly_detection_response_1, anomaly_detection_response_2, anomaly_detection_response_3, row) in enumerate(zip(anomaly_detection_responses_1, anomaly_detection_responses_2, anomaly_detection_responses_3, df.iterrows())):
            _, row = row  # Unpack the row
            inputfile: str = row['inputfile']
            table_schema: str = row['table_schema']
            num_anomalies: str = row['num_anomalies']
            anomalies: str = row['anomalies']
            
            usage = 0
            latency = 0
            cost = 0
            anomaly_detection_response_result = ''

            if anomaly_detection_response_1[1] is not None:
                cost = self.util.calculate_cost(anomaly_detection_response_1[1], self.model_id)
                usage = json.dumps(anomaly_detection_response_1[1])
            
            if anomaly_detection_response_1[2] is not None:
                latency = anomaly_detection_response_1[2]
            
            if anomaly_detection_response_1 and anomaly_detection_response_1[0]:
                anomaly_detection_response_result = anomaly_detection_response_1[0]

            if self.prompt_template_2 != '':
                if anomaly_detection_response_2[1] is not None:
                    cost += self.util.calculate_cost(anomaly_detection_response_2[1], self.model_id)
                    usage += json.dumps(anomaly_detection_response_2[1])
                
                if anomaly_detection_response_2[2] is not None:
                    latency += anomaly_detection_response_2[2]
                
                if anomaly_detection_response_2 and anomaly_detection_response_2[0]:
                    anomaly_detection_response_result = anomaly_detection_response_2[0]

            if self.prompt_template_3 != '':
                if anomaly_detection_response_3[1] is not None:
                    cost += self.util.calculate_cost(anomaly_detection_response_3[1], self.model_id)
                    usage += json.dumps(anomaly_detection_response_3[1])
                
                if anomaly_detection_response_3[2] is not None:
                    latency += anomaly_detection_response_3[2]

                if anomaly_detection_response_3 and anomaly_detection_response_3[0]:
                    anomaly_detection_response_result = anomaly_detection_response_3[0]

            
            # Create eval prompt
            prompt = self.build_grader_prompt(anomaly_detection_response=anomaly_detection_response_result, 
                                              table_schema=table_schema,
                                              ground_truth_anomalies=anomalies
                                            )
            
            # Parse eval results
            eval_result = self.eval_llm.generate(prompt, attachment_file=inputfile)
            classification = self.util.extract_with_regex(str(eval_result[0]), self.util.CLASSIFICATION_PATTERN)
            reasoning = self.util.extract_with_regex(str(eval_result[0]), self.util.REASONING_PATTERN)
            score = self.util.extract_with_regex(str(eval_result[0]), self.util.SCORE_PATTERN)
        
            
            # Create new record
            result = {
                'table_schema': table_schema,
                'inputfile': inputfile,
                'anomaly_detection_response': anomaly_detection_response_result,
                'groundtruth_num_anomalies': num_anomalies,
                'groundtruth_anomalies': anomalies,
                'score': score,
                'reasoning': reasoning,
                'classification': classification,
                'usage': usage,
                'latency': latency,
                'cost': cost,
            }
            results.append(result)
        
        new_dataframe = pd.DataFrame(results)
        return new_dataframe