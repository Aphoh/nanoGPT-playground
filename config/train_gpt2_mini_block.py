# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-gpt2-mini-block"
eval_interval = 1000  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often
weight_decay = 1e-1

wandb_log = True  # override via command line if you like
wandb_project = "gpt2-mini-block"
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0

# block linear stuff
block_linear = True
bias = False
mlp_ratio = 1
block_m = 8
block_n = 32
block_k = 16


learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
