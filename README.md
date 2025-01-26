# Gemma-2-2b: Reasoning "Enhancement" with CoT Data

**tl;dr:** This project explores enhancing a small language model's (**gemma-2-2b-4bit**) reasoning abilities by fine-tuning it with chain-of-thought (CoT) data generated using GPT-4o-mini. the goal: insights into cheap (compute) but capable model I can levereage for a RAG web app.

## motivation
* looking into efficient ways to buil a RAG web app for my final project.
* traditional NNs are not really applicable for tasks like these (I think), and LLM are expensive.
* this project investigates improving the reasoning capabilities of a small, efficient transformer using CoT prompting.

## data
* **source:** the project is buit on the [truthfulqa dataset](https://huggingface.co/datasets/truthfulqa/truthful_qa) on huggingface, a collection of **817 questions**  testing reasoning abilities. Just short question-answer pairs, no CoT answers.
* I used the [CAMEL-AI framework](https://github.com/camel-ai/camel) to generate more detailed, chain-of-thought (cot) reasoning answers.
    * I instantiated a chat agent using OpenAI's GPT-4o-Mini API, which automated the generation of CoT-answers for all 817 prompts (I had some money left in an openAI account).
* I created two CoT-augmented datasets: a smaller version (100 entries) and a larger version with 761 entries to test how important the size of the fine-tuning data is. These datasets are publicly [available on huggingface]([redacted link](https://huggingface.co/datasets/0fg)

## model & training

* **fine-tuning approach**: the project uses **Unsloth**, a library designed for more efficient fine-tuning of large LLM, on gemma-2-2b-4bit model.
* for training efficiency I used **LoRA** (low-rank adaptation) to significantly reduce the number of trainable parameters.
* **training data**: the project utilizes the two CoT-augmented truthfulqa datasets I created,
* **process**: the script first converts the CoT data into a alpaca instruction-following format required for supervised fine-tuning (SFT). Then, it uses huggingface's TrainingArguments to define hyperparameters (batch size, learning rate, optimizer etc.) before the training is initiated with the Huggingface transformer trainer.

## Evaluations

* **finetuning elasticity:** the experiments showed that the fine-tuning process was somewhat inelastic. Adjusting hyperparameters either resulted in minimal changes to the model's performance or quickly led to overfitting.

* **dataset size**: the larger CoT-augmented truthfulqa dataset with 761 entries demonstrated  slightly more resistance to overfitting compared to the smaller dataset with 100 entries.

* **benchmarking challenges**: attempts to evaluate the fine-tuned models using industry-standard benchmarks were unsuccessful. The only one I managed to get to work was [MMLU](https://docs.confident-ai.com/docs/benchmarks-mmlu) (massive multitask language understanding) – this evaluation consumed significant gpu compute resources and, annoyingly, resulted in a score of 0.0 for the fine-tuned models. This outcome, despite the models generating pretty coherent responses, probably means that MMLU is ill-suited for evaluating models specifically trained on CoT output.


