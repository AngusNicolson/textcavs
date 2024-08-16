
from pathlib import Path
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def main():
    max_response_length = 200
    idx = 0
    prompt_dir = Path("./data/prompts")
    class_names_long_path = "data/class_names.txt"
    with open(class_names_long_path, "r") as fp:
        class_names = fp.read()

    targets = class_names.split("\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-dpo-13b")
    model = AutoModelForCausalLM.from_pretrained("allenai/tulu-2-dpo-13b", quantization_config=bnb_config, device_map='cuda')

    with open(prompt_dir / "base.txt", "r") as fp:
        base_prompt = fp.read()

    prompts = ["seen_around", "part_of", "similar_to", "superclass"]

    for prompt in prompts:
        path = prompt_dir / f"{prompt}.txt"
        with open(path, "r") as fp:
            prompt_text = fp.read()

        responses = get_all_responses(
            base_prompt,
            prompt_text,
            targets,
            model,
            tokenizer,
            max_response_length=max_response_length,
            log=True
        )
        with open(f"data/text_concepts/tulu_4bit_{prompt}_{idx:02d}.json", "w") as fp:
            json.dump(responses, fp, indent=2)

    print("Done!")


def get_all_responses(base_prompt, prompt_template, targets, model, tokenizer, max_response_length=200, log=True):
    outputs = {}
    for target in targets:
        input_text = prompt_template.replace("{target}", target)
        prompt = base_prompt.replace('{msg}', input_text)
        generated_output = get_response_for_single_prompt(
            prompt,
            model,
            tokenizer,
            max_response_length=max_response_length
        )
        outputs[target] = generated_output
        if log:
            print(generated_output)
    return outputs


def get_response_for_single_prompt(prompt, model, tokenizer, max_response_length=200):
    with torch.no_grad():
        output = model.generate(
            **tokenizer(
                prompt,
                return_tensors="pt").to("cuda"),
            max_new_tokens=max_response_length
        )
    generated_output = tokenizer.decode(output[0])
    del output
    return generated_output


if __name__ == "__main__":
    main()
