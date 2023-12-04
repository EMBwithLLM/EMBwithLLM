import os
import argparse
import json
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

model_name = 'berkeley-nest/Starling-LM-7B-alpha'

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.max_new_tokens = 20
generation_config.temperature = 0.1
generation_config.do_sample = True

pipe = pipeline("text-generation", 
                model = model,
                tokenizer = tokenizer,
                return_full_text = True,
                generation_config = generation_config,
                num_return_sequences=1,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.eos_token_id
               )


def llm_completion(messages, max_trials: int = 1):
    
    output, error = None, None
    #print(pipe( **kwargs))
    
    for _ in range(max_trials):
        try:
            output = pipe(messages)[0]['generated_text']
            break
        except Exception as e:
            error = e
            pass
    
    return output, error


def prepare_data(instruction, datum):
    postfix = " Please respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]

    prompt = f"GPT4 Correct User: {input_txt}\n\n{instruction}{postfix}<|end_of_turn|>GPT4 Correct Assistant:"
    
    return prompt


def post_process(completion, choices):
    content = completion.split("Assistant:")[-1]
    #content = completion.split("### Assistant:\n")[-1]
    
    result = []
    for choice in choices:
        choice_txt = "Choice" + choice
        if choice_txt in content:
            result.append(choice.strip())
           
    print(f"content:{content}, result : {result}")
    
    return content, result


def predict(args):
#     relative_path = "/predicted_triplet_results"
#     directory_path = os.path.join(os.getcwd(), relative_path)
#     os.makedirs(directory_path , exist_ok=True)
#     print(directory_path)
    pred_path = args.data_path.split("/")[-1].replace(".json", f"-{args.model_name}{'-temp' + str(round(args.temperature, 1)) if args.temperature > 0 else ''}-pred.json")
    #pred_path = f"{MODEL_NAME}-pred.json"
    pred_path = os.path.join("predicted_triplet_results", pred_path)
    print("Save in: ", pred_path)
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
    
    with open("prompts.json", 'r') as f:
        prompts = json.load(f)
        task_prompt = prompts[args.dataset]
    
    for d in data:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)
    
    for idx, datum in tqdm(enumerate(data), total=len(data)):
        #if idx == 0:
            #print(datum['prepared'])
        # breakpoint()
        if 'prediction' in datum:
            continue
#         messages = [
#             {"role": "user", "content": datum['prepared']}
#         ]
        messages = datum['prepared']

        completion, error = llm_completion(messages, max_trials=args.max_trials)
        
        
        if completion is None:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, 'w') as f:
                json.dump(data, f)
            
            breakpoint()
        else:
            content, results = post_process(completion, datum['options'])
            data[idx]['content'] = content
            data[idx]['prediction'] =  results

            # breakpoint()
        
        if idx % args.save_every == 0 and idx > 0:
            print(f"Saving data after {idx + 1} inference.")
            with open(pred_path, "w") as f:
                json.dump(data, f)
        
        # if idx > 10:
        #     break
    
    with open(pred_path, "w") as f:
        json.dump(data, f)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--model_name", type=str, default='starling')
    parser.add_argument("--max_trials", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    predict(args)