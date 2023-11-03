

train_hooks = dict(
    post_epoch=[
        dict(type="CheckPointsHook", epoch_every=10)
    ]
)