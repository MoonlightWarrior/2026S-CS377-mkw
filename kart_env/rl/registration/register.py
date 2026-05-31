from ..action_parser import DiscreteLookupAction
from ..obs_builder import KartVectorObs, TeamVectorObs
from ..reward_function import (
    FinishReward,
    ForwardVelocityReward,
    ItemUseReward,
    LapCompletionReward,
    MTBoostReward,
    MTChargeReward,
    MushroomBoostActiveReward,
    PositionImprovementReward,
    RacePositionReward,
    RaceProgressReward,
    SMTChargeReward,
    SmoothDrivingReward,
    SpeedReward,
    TeamProgressReward,
)
from .action_parser import register_action_parser
from .obs_builder import register_obs_builder
from .reward_function import register_reward_function


def register_obs_builders() -> None:
    register_obs_builder("KartVectorObs", KartVectorObs)
    register_obs_builder("TeamVectorObs", TeamVectorObs)


def register_action_parsers() -> None:
    register_action_parser("DiscreteLookupAction", DiscreteLookupAction)


def register_reward_functions() -> None:
    register_reward_function("RaceProgressReward", RaceProgressReward)
    register_reward_function("SpeedReward", SpeedReward)
    register_reward_function("FinishReward", FinishReward)
    register_reward_function("RacePositionReward", RacePositionReward)
    register_reward_function("MTBoostReward", MTBoostReward)
    register_reward_function("SmoothDrivingReward", SmoothDrivingReward)
    register_reward_function("TeamProgressReward", TeamProgressReward)
    register_reward_function("LapCompletionReward", LapCompletionReward)
    register_reward_function("ItemUseReward", ItemUseReward)
    register_reward_function("ForwardVelocityReward", ForwardVelocityReward)
    register_reward_function("MTChargeReward", MTChargeReward)
    register_reward_function("SMTChargeReward", SMTChargeReward)
    register_reward_function("PositionImprovementReward", PositionImprovementReward)
    register_reward_function("MushroomBoostActiveReward", MushroomBoostActiveReward)


def register_components() -> None:
    register_obs_builders()
    register_action_parsers()
    register_reward_functions()
