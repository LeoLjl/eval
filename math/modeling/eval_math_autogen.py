import os
import openai
import numpy as np
import time
import json
import random
import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from dataset.util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv

config_list = [
    {
        # config list here
    }
]

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 8,
    jitter: bool = True,
    max_retries: int = 20,
    errors: tuple = (openai.error.RateLimitError, openai.error.APIConnectionError, openai.error.APIError, openai.error.ServiceUnavailableError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                print("<error>", e, "</error>")

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= min(exponential_base * (1 + jitter * random.random()), max_delay)

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

# @retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(20))
@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def call_engine(train_prompt, problem, engine="gpt-3.5-turbo"):
    '''
    Given a problem, returns the most likely answer determined by the GPT engine 
    '''
    mathproxyagent = MathUserProxyAgent(
        name="mathproxyagent", 
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )
    assistant = autogen.AssistantAgent(
        name="assistant", 
        system_message="You are a helpful assistant.",
        llm_config={
            "timeout": 600,
            "seed": 10,
            "config_list": config_list,
        }
    )
    print("Problem: ", problem)
    mathproxyagent.initiate_chat(assistant, problem=problem)
    response = mathproxyagent.chat_messages[assistant][-1]["content"]
    return response

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
with open("prompt/zero_shot_cot.txt", "r") as f:
    train_prompt = f.read()
rootdir = "MATH/test"


def run(engine="davinci", max=-1):
    outputs = []
    answers = []
    types = []
    levels = []

    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fnames_list.append(os.path.join(subdir, file))
            with open(os.path.join(subdir, file), 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {file}", e)
                    raise e
                prob_level = problem_data["level"]
                prob_type = problem_data["type"]
                try:
                    prob_level = int(prob_level.split("Level ")[1])
                except:
                    prob_level = None
                model_output = call_engine(train_prompt, problem_data["problem"], engine=engine)
                model_answer = remove_boxed(last_boxed_only_string(model_output))
                answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))

                levels.append(prob_level)
                types.append(prob_type)
                outputs.append(model_output)
                answers.append(answer)

                print("Model answer:")
                print(model_answer)
                print("Correct answer:")
                print(answer)
                print("--------------------------------------------")

                try:
                    equiv = is_equiv(model_answer, answer)
                except:
                    equiv = False
                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(equiv)
                else:
                    cors[(prob_level, prob_type)] = [equiv]
                if prob_level in level_cors:
                    level_cors[prob_level].append(equiv)
                else:
                    if prob_level is not None:
                        level_cors[prob_level] = [equiv]
                if prob_type in subject_cors:
                    subject_cors[prob_type].append(equiv)
                else:
                    if prob_type is not None:
                        subject_cors[prob_type] = [equiv]
                if equiv:
                    correct += 1
                total += 1

                print(str(correct) + "/" + str(total))

            if max > 0 and total > max:
                break
        if max > 0 and total > max:
            break

    with open("outputs_answers_gpt35_{}.txt".format(engine), "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))

if __name__ == "__main__":
    # engines = ["davinci", "curie", "babbage", "ada"][::-1]
    # for engine in engines:
    #     run(engine)
    engine = "gpt-35-turbo"
    run(engine, max=10)
    # for testing:
    # for engine in ["ada"]:
    #     run(engine, max=10)
