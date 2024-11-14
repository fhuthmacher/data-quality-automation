
from awsglue import DynamicFrame
import pyspark.sql.functions as F
import datetime
import time

from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, json
import time
from botocore.config import Config
import base64
from io import BytesIO
import base64
import pandas as pd
# from PIL import Image

class BedrockLLMWrapper():
    def __init__(self,
        model_id: str = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        embedding_model_id: str = 'amazon.titan-embed-image-v1',
        system_prompt: str = 'You are a helpful AI Assistant.',
        region: str = 'us-east-1',
        top_k: int = 5,
        top_p: int = 0.7,
        temperature: float = 0.0,
        max_token_count: int = 4000,
        max_attempts: int = 3,
        debug: bool = False

    ):

        
        
        self.embedding_model_id = embedding_model_id
        self.system_prompt = system_prompt
        self.region = region
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.max_attempts = max_attempts
        self.debug = debug
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )

        self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=config, region_name=self.region)

        self.model_id = model_id

    def get_valid_format(self, file_format):
        format_mapping = {
            'jpg': 'jpeg',
            'gif': 'gif',
            'png': 'png',
            'webp': 'webp'
        }
        return format_mapping.get(file_format.lower(), 'jpeg')  # Default to 'jpeg' if format is not recognized
    
    # def process_image(self, image_path, max_size=(512, 512)):
    #     with open(image_path, "rb") as image_file:
    #         # Read the image file
    #         image = image_file.read()
    #         image = Image.open(BytesIO(image)).convert("RGB")
            
    #         # Resize image while maintaining aspect ratio
    #         image.thumbnail(max_size, Image.LANCZOS)
            
    #         # Create a new image with the target size and paste the resized image
    #         new_image = Image.new("RGB", max_size, (255, 255, 255))
    #         new_image.paste(image, ((max_size[0] - image.size[0]) // 2,
    #                                 (max_size[1] - image.size[1]) // 2))
            
    #         # Save to BytesIO object
    #         buffered = BytesIO()
    #         new_image.save(buffered, format="JPEG")
            
    #         # Encode to base64
    #         input_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf8')
        
    #     return input_image_base64

    def get_embedding(self, input_text=None, image_path=None):
        """
        This function is used to generate the embeddings for a specific chunk of text
        """
        accept = 'application/json'
        contentType = 'application/json'
        request_body = {}

        if input_text:
            request_body["inputText"] = input_text
        if image_path:
            # Process and encode the image
            img_base64 = '' #self.process_image(image_path)
            request_body["inputImage"] = img_base64

        # request_body["dimensions"] = 1024
        # request_body["normalize"] = True

        if 'amazon' in self.embedding_model_id:
            embeddingInput = json.dumps(request_body)
            response = self.bedrock_runtime.invoke_model(body=embeddingInput, 
                                                        modelId=self.embedding_model_id, 
                                                        accept=accept, 
                                                        contentType=contentType)
            embeddingVector = json.loads(response['body'].read().decode('utf8'))
            return embeddingVector['embedding']
                
        if 'cohere' in self.embedding_model_id:
            request_body["input_type"] = "search_document" # |search_query|classification|clustering
            request_body["truncate"] = "NONE" # NONE|START|END
            embeddingInput = json.dumps(request_body)
    
            response = self.bedrock_runtime.invoke_model(body=embeddingInput, 
                                                            modelId=self.embedding_model_id, 
                                                            accept=accept, 
                                                            contentType=contentType)
    
            response_body = json.loads(response.get('body').read())
            # print(response_body)
            embeddingVector = response_body['embedding']
            
            return embeddingVector
    
    def generate(self,prompt,attachment_file=None, image_file=None, image_file2=None):
        if self.debug: 
            print('entered BedrockLLMWrapper generate')
        message = {}
        attempt = 1
        if image_file is not None:
            if self.debug: 
                print('processing image1: ', image_file)
            # extract file format from the image file
            file_format = image_file.split('.')[-1]
            valid_format = self.get_valid_format(file_format)

            # Open and read the image file
            with open(image_file, 'rb') as img_file:
                image_bytes = img_file.read()
                if self.debug: 
                    print('image_bytes: ', image_bytes)
                    print('valid_format: ', valid_format)

            message = {
                "role": "user",
                "content": [
                    { "text": "Image 1:" },
                    {
                        "image": {
                            "format": valid_format,
                            "source": {
                                "bytes": image_bytes 
                            }
                        }
                    },
                    { "text": prompt }
                ],
                    }
            
        if image_file is not None and image_file2 is not None:
            if self.debug: 
                print('processing image2: ', image_file2)
            # extract file format from the image file
            file_format2 = image_file2.split('.')[-1]
            valid_format2 = self.get_valid_format(file_format2)

            with open(image_file2, 'rb') as img_file:
                image_bytes2 = img_file.read()
                if self.debug: 
                    print('image_bytes2: ', image_bytes2)
                    print('valid_format2: ', valid_format2)
            
            message = {
            "role": "user",
            "content": [
                { "text": "Image 1:" },
                {
                    "image": {
                        "format": valid_format,
                        "source": {
                            "bytes": image_bytes 
                        }
                    }
                },
                { "text": "Image 2:" },
                {
                    "image": {
                        "format": valid_format2,
                        "source": {
                            "bytes": image_bytes2 
                        }
                    }
                },
                { "text": prompt }
            ],
                }
        
        if attachment_file is not None:
            with open(attachment_file, 'rb') as attachment_file:
                attachment_bytes = attachment_file.read()
                if self.debug: 
                    print('attachment_bytes: ', attachment_bytes)
            
            message = {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "name": "Document 1",
                            "format": "csv",
                            "source": {
                                "bytes": attachment_bytes
                            }
                        }
                    },
                    { "text": prompt }
                ]
            }
            
        if image_file is None and image_file2 is None and attachment_file is None:
            message = {
                "role": "user",
                "content": [{"text": prompt}]
            }
        messages = []
        messages.append(message)
        
        # model specific inference parameters to use.
        if "anthropic" in self.model_id.lower():
            system_prompts = [{"text": self.system_prompt}]
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                                "stopSequences": ["\n\nHuman:"],
                                "topP": self.top_p,
                            }
            additional_model_fields = {"top_k": self.top_k}
        else:
            system_prompts = []
            # Base inference parameters to use.
            inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                            }
            additional_model_fields = {"top_k": self.top_k}

        if self.debug: 
            print('Sending: System: ',system_prompts,'Messages: ',str(messages))

        while True:
            try:

                # Send the message.
                response = self.bedrock_runtime.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )

                text = response['output'].get('message').get('content')[0].get('text')
                usage = response['usage']
                latency = response['metrics'].get('latencyMs')

                if self.debug: 
                    print(f'text: {text} ; and token usage: {usage} ; and query_time: {latency}')    
                
                break
               
            except Exception as e:
                print("Error with calling Bedrock: "+str(e))
                attempt+=1
                if attempt>self.max_attempts:
                    print("Max attempts reached!")
                    result_text = str(e)
                    break
                else:#retry in 10 seconds
                    print("retry")
                    time.sleep(60)

        # return result_text
        return [text,usage,latency]

     # Threaded function for queue processing.
    def thread_request(self, q, results):
        while True:
            try:
                index, prompt = q.get(block=False)
                data = self.generate(prompt)
                results[index] = data
            except Queue.Empty:
                break
            except Exception as e:
                print(f'Error with prompt: {str(e)}')
                results[index] = str(e)
            finally:
                q.task_done()

 
    def generate_threaded(self, prompts, attachments=None, images=None, max_workers=15):
        
        if images is None:
            images = [None] * len(prompts)
        elif len(prompts) != len(images):
            raise ValueError("The number of prompts must match the number of images (or images must be None)")
        
        if attachments is None:
            attachments = [None] * len(prompts)
        elif len(prompts) != len(attachments):
            raise ValueError("The number of prompts must match the number of attachments (or attachments must be None)")

        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            future_to_index = {executor.submit(self.generate, prompt, attachment_file, image_file): i 
                               for i, (prompt, attachment_file, image_file) in enumerate(zip(prompts, attachments, images))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    results[index] = str(exc)
        
        return results

PROMPT_TEMPLATE_FIX ="""
You are an expert data analyst specializing in data quality and anomaly detection. 
Your task is to analyze the below data quality anomaly detection result and fix the data quality issues row by row.

Data quality result:
{DATA_QUALITY_RESULT}

Target table schema:
{TARGET_TABLE_SCHEMA}

Please analyze the data thoroughly.

Return the response in the following JSON format, ensuring that all special characters
are properly escaped and the JSON iswell-formed:

[
  {{"column_name1": "column_value1", 
  "column_name2": "column_value2", 
  "column_name3": "column_value3",
  <all fields from  source data> 
  }},
  {{<...>}},
]

Do not include the data quality result related columns DataQualityRulesPass, DataQualityRulesFail, DataQualityRulesSkip, DataQualityEvaluationResult.
Only include the columns that are in the target table schema.
Only include JSON and nothing else in the response"""


def bedrock_dq_fix(self, llm=None, prompt_template=None):
  if not llm:
      # default to claude sonnet 3.5
      llm = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
  
  if not prompt_template:
      # default to the prompt template for fixing data quality issues
      prompt_template = PROMPT_TEMPLATE_FIX

  rowLevelOutcomes_df = self.toDF()

  # check if there are any data quality issues

  # show all data
  rowLevelOutcomes_df.show(truncate=False)
  
  # filter only the Passed records
  rowLevelOutcomes_df_passed = rowLevelOutcomes_df.filter(rowLevelOutcomes_df.DataQualityEvaluationResult == "Passed") # Filter only the Passed records.
  
  # filter only the Failed records
  rowLevelOutcomes_df_error = rowLevelOutcomes_df.filter(rowLevelOutcomes_df.DataQualityEvaluationResult == "Failed")

  # show the failed records
  rowLevelOutcomes_df.filter(rowLevelOutcomes_df.DataQualityEvaluationResult == "Failed").show(5, truncate=False) # Review the Failed records                    

  if rowLevelOutcomes_df_error.count() == 0:
    print("No data quality issues found")
    rowLevelOutcomes_final_df = DynamicFrame.fromDF(rowLevelOutcomes_df, self.glue_ctx, "dq_bedrock_data")
    return rowLevelOutcomes_final_df
  
  # convert frame1 (dynamic frame) to a string
  frame1_str = rowLevelOutcomes_df.toPandas().to_string()
  print(f'Data Quality Result:{frame1_str}')

  # convert frame2 (dynamic frame) to a string
  source_reference_spark_df = self.glue_ctx.spark_session.table("source_reference_schema")

  # get the schema of the target table
  target_table_schema = source_reference_spark_df.schema

  frame2_str = source_reference_spark_df.toPandas().to_string()
  print(f'Data Sample from target table:{frame2_str}')

  prompt = prompt_template.format(DATA_QUALITY_RESULT=frame1_str, TARGET_TABLE_SCHEMA=frame2_str)
  print(f'Prompt: {prompt}')

  bedrock = BedrockLLMWrapper(debug=False, max_token_count=4096, model_id=llm)
  print(f'Calling Bedrock to generate response')
  result = bedrock.generate(prompt)
  print(f'Got a LLM response: {result}')

  # parse json result[0] to a dynamic dataframe
  json_data = json.loads(result[0])
  pandas_df = pd.DataFrame(json_data)
  
  # Get schema from input DynamicFrame
  input_schema = self.schema()
  print(f'Input schema: {str(input_schema)}')

  spark_df = self.glue_ctx.spark_session.createDataFrame(pandas_df,schema=target_table_schema)
  df = DynamicFrame.fromDF(spark_df, self.glue_ctx, "dq_bedrock_data")

  # pretty print the dynamic dataframe
  print(df.printSchema())

  return df
  
# register the function as a method of DynamicFrame
DynamicFrame.bedrock_dq_fix = bedrock_dq_fix