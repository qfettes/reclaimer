from gym.envs.registration import register

register(
    id='dsb-social-media-v0',
    entry_point='gym_dsb.envs:DSBSocialMediaService',
    nondeterministic = True
)