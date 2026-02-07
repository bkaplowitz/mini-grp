

## PaliGemma3B-based VLA model (image + goal image + text -> continuous actions)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class PaliGemmaVLA(nn.Module):
    """
    VLA model using PaliGemma backbone with a separate action head.
    It fuses text + concatenated current/goal images, then predicts continuous actions.
    """
    def __init__(self, cfg, model_name='google/paligemma-3b-pt-224', freeze_backbone=True):
        super().__init__()
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.backbone = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size * 2, cfg.action_bins),
        )

    def _prepare_images(self, images, goal_images):
        # Convert [-1, 1] float tensors to uint8 HWC and concatenate along width.
        images_u8 = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        goals_u8 = ((goal_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images_u8 = images_u8.permute(0, 2, 3, 1).cpu().numpy()
        goals_u8 = goals_u8.permute(0, 2, 3, 1).cpu().numpy()
        combined = [np.concatenate([img, goal], axis=1) for img, goal in zip(images_u8, goals_u8)]
        return combined

    def forward(self, images, goal_texts, goal_images, targets=None):
        combined_images = self._prepare_images(images, goal_images)
        inputs = self.processor(
            text=goal_texts,
            images=combined_images,
            return_tensors='pt',
            padding=True,
        ).to(self.backbone.device)
        outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden)
        pooled = last_hidden[:, -1, :]
        action_predictions = self.action_head(pooled)
        loss = None
        if targets is not None:
            loss = F.mse_loss(action_predictions, targets)
        return action_predictions, loss

def create_paligemma_vla_model(cfg, device='cuda', freeze_backbone=True):
    if not hasattr(cfg, 'max_text_length'):
        cfg.max_text_length = 64
    model = PaliGemmaVLA(
        cfg=cfg,
        model_name='google/paligemma-3b-pt-224',
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    return model

def train_paligemma_vla_model(model, dataset, cfg, device='cuda'):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.learning_rate),
    )
    for iter in range(cfg.max_iters):
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            model.eval()
            with torch.no_grad():
                indices = np.random.choice(len(dataset["img"]), 4, replace=False)
                val_images = dataset["image_enc"][indices].permute(0, 3, 1, 2)
                val_goal_texts = [dataset["goal"][i] for i in indices]
                val_goal_images = dataset["goal_image_enc"][indices].permute(0, 3, 1, 2)
                val_actions = dataset["action_enc"][indices]
                _, val_loss = model(val_images, val_goal_texts, val_goal_images, val_actions)
                print(f"step {iter}: val loss {val_loss.item():.4f}")
            model.train()
        indices = np.random.choice(len(dataset["img"]), 4, replace=False)
        images = dataset["image_enc"][indices].permute(0, 3, 1, 2)
        goal_texts = [dataset["goal"][i] for i in indices]
        goal_images = dataset["goal_image_enc"][indices].permute(0, 3, 1, 2)
        actions = dataset["action_enc"][indices]
        predictions, loss = model(images, goal_texts, goal_images, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if iter % 100 == 0:
            print(f"step {iter}: train loss {loss.item():.4f}")
    return model

print("PaliGemma VLA utilities defined")



## Grab a chunk of data for training
import tensorflow_datasets as tfds
import cv2
import numpy as np
image_shape = [64, 64, 3]
num_episodes = 20 ## How many episodes to grab from the dataset for training

builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
datasetRemote = builder.as_dataset(split='train[:' + str(num_episodes) + ']')
dataset = {"img": [], "action": [], "goal": [], "goal_img": [],
                "rotation_delta": [], "open_gripper": [] }
shortest_goal_txt = 10000000000
for episode in datasetRemote:
    episode_ = {'steps': [] }
    episode = list(episode['steps'])
    ## Goal image is just the last image/state/observation in the episode
    goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))
    for i in range(len(episode)):
        obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))
        goal = episode[i]['observation']['natural_language_instruction'].numpy().decode()
        dataset["img"].append(obs)
        dataset["action"].append(np.array(np.concatenate((episode[i]['action']['world_vector'], 
                                                          episode[i]['action']['rotation_delta'],
                                                        [episode[i]['action']['open_gripper']]), axis=0)))
         
        dataset["rotation_delta"].append(np.array(episode[i]['action']['rotation_delta']))
        dataset["open_gripper"].append(np.array(episode[i]['action']['open_gripper']))
        dataset["goal"].append(goal)
        dataset["goal_img"].append(goal_img)
        if len(goal) < shortest_goal_txt: shortest_goal_txt = len(goal)

# here are all the unique characters that occur in this text
chars = sorted(list(set([item for row in dataset["goal"] for item in row])))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode_txt = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode_txy = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print("vocab_size:", vocab_size)
print("example text encode:", encode_txt(dataset["goal"][0]))

print("Dataset shape:", len(dataset["img"]))
dataset["img"] = np.array(dataset["img"], dtype=np.uint8)
dataset["action"] = np.array(dataset["action"], dtype=np.float32)
# dataset["goal"] = np.array(encode_txt(dataset["goal"]), dtype=np.float32)
dataset["goal_img"] = np.array(dataset["goal_img"], dtype=np.uint8)


## Example: Create and test VLA model
## Uncomment to run VLA instead of GRP


# Create VLA model
from box import Box
import yaml

# Load config
with open('./mini-grp/conf/config.yaml', 'r') as f:
    cfg_dict = yaml.safe_load(f)    
cfg = Box(cfg_dict)

# Prepare data encodings
a_std, a_mean = (dataset["action"].std(axis=0) + 0.001) * 1.5, dataset["action"].mean(axis=0)
cfg.action_bins = len(a_mean)
encode_action = lambda af: (((af - a_mean)/(a_std))).astype(np.float32)
encode_state = lambda af: ((af/(255.0)*2.0)-1.0).astype(np.float32)

# Encode dataset for VLA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset["image_enc"] = torch.tensor(encode_state(dataset["img"])).to(device)
dataset["goal_image_enc"] = torch.tensor(encode_state(dataset["goal_img"])).to(device)
dataset["action_enc"] = torch.tensor(encode_action(dataset["action"]), dtype=torch.float).to(device)

# Initialize VLA model
vla_model = create_paligemma_vla_model(cfg, device=device)

# Test forward pass with a small batch
test_batch_size = 4
test_indices = np.random.choice(len(dataset["img"]), test_batch_size, replace=False)
test_images = dataset["image_enc"][test_indices].permute(0, 3, 1, 2)
test_goal_texts = [dataset["goal"][i] for i in test_indices]
test_goal_images = dataset["goal_image_enc"][test_indices].permute(0, 3, 1, 2)
test_actions = dataset["action_enc"][test_indices]

print(f"Test batch shapes:")
print(f"  Images: {test_images.shape}")
print(f"  Goal texts: {len(test_goal_texts)} strings")
print(f"  Goal images: {test_goal_images.shape}")
print(f"  Actions: {test_actions.shape}")

# Forward pass
predictions, loss = vla_model(test_images, test_goal_texts, test_goal_images, test_actions)
print(f"  VLA Model forward pass successful!")
print(f"  Predictions shape: {predictions.shape}")
print(f"  Loss: {loss.item():.4f}")

# Train VLA model (uncomment to train)
vla_model = train_paligemma_vla_model(vla_model, dataset, cfg, device=device)

print("VLA example code ready (uncomment to run)")