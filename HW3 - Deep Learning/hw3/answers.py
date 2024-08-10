r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 250
    hypers['seq_len'] = 100
    hypers['h_dim'] = 250
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.01
    hypers['learn_rate'] = 0.005
    hypers['lr_sched_factor'] = 0.05
    hypers['lr_sched_patience'] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "When she was just a girl"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""

We split the corpus into smaller sequences because loading the entire text at once would use up a lot of memory, slowing down the training. By breaking it into smaller parts, we can load and process each piece one at a time, making the training faster and easier on the computer. Plus, working with smaller chunks helps the model learn better from the data.

"""

part1_q2 = r"""

The generated text can show memory longer than the sequence length because the RNN's hidden state learns how to predict the next word based on the sequence it was trained on. Even though we train it on shorter sequences, the hidden state picks up on patterns and connections across different sequences, allowing the network to remember and generalize information from the entire text.

"""

part1_q3 = r"""

We don’t shuffle the order of batches during training because we want to keep the sentences in the correct order. By training in the right sequence, the model can maintain the logical flow and relationships between the sentences. This also helps the model understand the context better, leading to text generation that’s more similar to the original text.

"""

part1_q4 = r"""

1) We lower the temperature to make the model’s predictions more focused and less random. If the predictions were too random, the results would be unpredictable and not very useful.

2) When the temperature is very high, the model's output becomes more random, like a uniform distribution where every option has almost the same chance of being picked. This leads to less meaningful and more chaotic text generation.

3) When the temperature is very low, the model becomes very certain about its choices, picking only the options it's most confident in. This can make the text repetitive and limited, as it avoids taking risks with less likely options.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers



part4_q1 = r"""

The fine-tuned DistilBERT models did better than the encoder we trained from scratch earlier. Method 2, where we retrained all parts of the model, got 86.9% accuracy, while Method 1, where we only trained the last few layers, got 65.4%. Both beat our custom encoder.
This happened because DistilBERT already knows a lot about language from its pre-training. When we fine-tune it, it can use this knowledge and adapt it to our specific task. It's like giving the model a head start.
But this won't always be true for every task. Sometimes, a model built from scratch might work better, especially for very specific or unusual tasks. It depends on things like how similar the task is to what DistilBERT was originally trained on, and how much data we have.

Method 2 did much better than Method 1, showing that letting the whole model adapt to our task was really helpful in this case.

"""

part4_q2 = r"""

If we froze the last layers and only fine-tuned internal parts like the multi-headed attention block, the model would probably do worse at our task. Here's why:
The last layers are really important for adapting to specific tasks like sentiment analysis. They're like the final decision-makers. If we freeze them, we're not letting the model adjust its final output for our task.
The internal layers, like the attention blocks, are more about general language understanding. Updating only these might help the model understand language better overall, but it wouldn't be able to use this understanding for our specific task.
It's kind of like upgrading a car's engine but not touching the steering wheel. The car might run better, but you can't steer it to where you want to go.

Also, if we change the internal parts but keep the last layers the same, they might not work well together anymore. It could be like trying to fit a new engine into an old car body - things might not line up right.
So, for our sentiment analysis task, it's usually better to fine-tune the later layers. This lets the model use what it already knows about language while learning to make the right decisions for our specific task.

"""

# ==============
