import gymnasium as gym
import numpy as np
import pytest
import torch as t
from einops import rearrange
from minigrid.wrappers import (
    ImgObsWrapper,
    OneHotPartialObsWrapper,
    RGBImgPartialObsWrapper,
)

from src.config import EnvironmentConfig, TransformerModelConfig
from src.environments.wrappers import RenderResizeWrapper, ViewSizeWrapper
from src.models.trajectory_transformer import (
    ActorTransformer,
    CloneTransformer,
    DecisionTransformer,
)

from transformer_lens.components import MLP, GatedMLP


def test_decision_transformer__init__():
    # test default values
    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        one_hot_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    transformer = decision_transformer.transformer
    # d_model: int = 128,

    # n_heads: int = 4,
    # d_mlp: int = 256,
    assert transformer.blocks[0].mlp.W_in.shape == (128, 256)
    assert transformer.blocks[0].mlp.W_out.shape == (256, 128)
    assert transformer.blocks[0].mlp.b_in.shape == (256,)
    assert transformer.blocks[0].mlp.b_out.shape == (128,)

    # n_layers: int = 2,
    assert len(decision_transformer.transformer.blocks) == 2
    # n_ctx: int = 2,
    assert transformer.blocks[0].attn.W_Q.shape == (4, 128, 32)
    assert transformer.blocks[0].attn.W_K.shape == (4, 128, 32)
    assert transformer.blocks[0].attn.W_V.shape == (4, 128, 32)
    assert transformer.blocks[0].attn.W_O.shape == (4, 32, 128)
    # layer_norm: str | None = None,
    assert isinstance(transformer.blocks[0].ln1, t.nn.Identity)
    # gated_mlp: bool = False,
    assert isinstance(transformer.blocks[0].mlp, MLP)
    # activation_fn: str = "relu",
    assert transformer.cfg.act_fn == "relu"

    num_params = sum(p.numel() for p in decision_transformer.parameters())
    assert num_params == 646364  # that's closer to being reasonable.

    num_params = sum(p.numel() for p in transformer.parameters())
    assert num_params == 264192  # that's closer to being reasonable.


def test_decision_transformer__init__2():
    # test non-default values
    transformer_config = TransformerModelConfig(
        d_model=256,
        n_heads=8,
        d_mlp=512,
        n_layers=2,
        n_ctx=5,
        layer_norm="LNPre",
        gated_mlp=True,
        activation_fn="gelu",
    )
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        one_hot_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    transformer = decision_transformer.transformer
    # d_model: int = 128,

    # n_heads: int = 2,
    # d_mlp: int = 256,
    assert transformer.blocks[0].mlp.W_in.shape == (256, 512)
    assert transformer.blocks[0].mlp.W_out.shape == (512, 256)
    assert transformer.blocks[0].mlp.b_in.shape == (512,)
    assert transformer.blocks[0].mlp.b_out.shape == (256,)

    # n_layers: int = 2,
    assert len(decision_transformer.transformer.blocks) == 2
    # n_ctx: int = 2,
    assert transformer.blocks[0].attn.W_Q.shape == (8, 256, 32)
    assert transformer.blocks[0].attn.W_K.shape == (8, 256, 32)
    assert transformer.blocks[0].attn.W_V.shape == (8, 256, 32)
    assert transformer.blocks[0].attn.W_O.shape == (8, 32, 256)
    # layer_norm: not
    assert not isinstance(transformer.blocks[0].ln1, t.nn.Identity)
    # gated_mlp: bool = True
    assert isinstance(transformer.blocks[0].mlp, GatedMLP)
    # activation_fn: str = "gelu",
    assert transformer.cfg.act_fn == "gelu"


def test_decision_transformer_img_obs_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor(obs).unsqueeze(0).unsqueeze(0)  # add block, add batch
    # actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=None, rtgs=rewards, timesteps=timesteps
    )

    assert (
        state_preds is None
    )  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert (
        reward_preds is None
    )  # no action or reward preds if no actions are given


@pytest.mark.parametrize("state_emb_type", ["grid", "cnn", "vit"])
def test_decision_transformer_grid_obs_forward(state_emb_type):
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(
        state_embedding_type=state_emb_type
    )
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=None, rtgs=rewards, timesteps=timesteps
    )

    assert (
        state_preds is None
    )  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert (
        reward_preds is None
    )  # no action or reward preds if no actions are given


def test_decision_transformer_grid_one_hot_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = OneHotPartialObsWrapper(env)
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        one_hot_obs=True,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=None, rtgs=rewards, timesteps=timesteps
    )

    assert (
        state_preds is None
    )  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert (
        reward_preds is None
    )  # no action or reward preds if no actions are given


def test_decision_transformer_view_size_change_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = ViewSizeWrapper(env, agent_view_size=3)
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        view_size=3,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=None, rtgs=rewards, timesteps=timesteps
    )

    assert (
        state_preds is None
    )  # no action or reward preds if no actions are given
    assert action_preds.shape == (1, 1, 7)
    assert (
        reward_preds is None
    )  # no action or reward preds if no actions are given


def test_decision_transformer_grid_obs_no_action_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig()
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    # actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=None, rtgs=rewards, timesteps=timesteps
    )

    assert state_preds is None
    assert action_preds.shape == (1, 1, 7)
    assert reward_preds is None


def test_decision_transformer_grid_obs_one_fewer_action_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=5)
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    decision_transformer = DecisionTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert decision_transformer is not None

    # our model should have the following:
    assert decision_transformer.state_embedding is not None
    assert decision_transformer.reward_embedding is not None
    assert decision_transformer.action_embedding is not None
    assert decision_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert decision_transformer.transformer is not None
    assert (
        type(decision_transformer.transformer).__name__ == "HookedTransformer"
    )

    # a linear layer to predict the next action
    assert decision_transformer.action_predictor is not None

    # a linear layer to predict the next reward
    assert decision_transformer.reward_predictor is not None

    # a linear layer to predict the next state
    assert decision_transformer.state_predictor is not None

    states = t.tensor([obs["image"], obs["image"]]).unsqueeze(
        0
    )  # add block, add batch
    actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    rewards = t.tensor([[0], [0]]).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([[0], [1]]).unsqueeze(0)  # add block, add batch

    state_preds, action_preds, reward_preds = decision_transformer.forward(
        states=states, actions=actions, rtgs=rewards, timesteps=timesteps
    )

    assert state_preds.shape == (1, 2, 147)
    assert action_preds.shape == (1, 2, 7)
    assert reward_preds.shape == (1, 2, 1)


def test_clone_transformer_grid_obs_no_action_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=1)
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    clone_transformer = CloneTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert clone_transformer is not None

    # our model should have the following:
    assert clone_transformer.state_embedding is not None
    assert clone_transformer.action_embedding is not None
    assert clone_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert clone_transformer.transformer is not None
    assert type(clone_transformer.transformer).__name__ == "HookedTransformer"

    # a linear layer to predict the next action
    assert clone_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert clone_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    _, action_preds = clone_transformer.forward(
        states=states, actions=None, timesteps=timesteps
    )

    assert action_preds.shape == (1, 1, 7)


def test_clone_transformer_grid_obs_one_fewer_action_forward():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=7)
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    clone_transformer = CloneTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert clone_transformer is not None

    # our model should have the following:
    assert clone_transformer.state_embedding is not None
    assert clone_transformer.action_embedding is not None
    assert clone_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert clone_transformer.transformer is not None
    assert type(clone_transformer.transformer).__name__ == "HookedTransformer"

    # a linear layer to predict the next action
    assert clone_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert clone_transformer.state_predictor is not None

    states = t.tensor([obs["image"], obs["image"]]).unsqueeze(
        0
    )  # add block, add batch
    actions = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    timesteps = t.tensor([[0], [1]]).unsqueeze(0)  # add block, add batch

    state_preds, action_preds = clone_transformer.forward(
        states=states, actions=actions, timesteps=timesteps
    )

    assert state_preds.shape == (1, 2, 147)
    assert action_preds.shape == (1, 2, 7)


def test_actor_transformer():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset()  # This now produces an RGB tensor only

    transformer_config = TransformerModelConfig(n_ctx=1)
    environment_config = EnvironmentConfig(
        env_id="MiniGrid-Empty-8x8-v0",
        img_obs=False,
    )

    actor_transformer = ActorTransformer(
        transformer_config=transformer_config,
        environment_config=environment_config,
    )

    assert actor_transformer is not None

    # our model should have the following:
    assert actor_transformer.state_embedding is not None
    assert actor_transformer.action_embedding is not None
    assert actor_transformer.time_embedding is not None

    # a GPT2-like transformer
    assert actor_transformer.transformer is not None
    assert type(actor_transformer.transformer).__name__ == "HookedTransformer"

    # a linear layer to predict the next action
    assert actor_transformer.action_predictor is not None

    # a linear layer to predict the next state
    assert actor_transformer.state_predictor is not None

    states = (
        t.tensor(obs["image"]).unsqueeze(0).unsqueeze(0)
    )  # add block, add batch
    # t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch
    actions = None
    timesteps = t.tensor([0]).unsqueeze(0).unsqueeze(0)  # add block, add batch

    action_preds = actor_transformer.forward(
        states=states, actions=actions, timesteps=timesteps
    )

    assert action_preds.shape == (1, 1, 7)
