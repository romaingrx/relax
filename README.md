> The purpose of this repository is to implement all the latest ML papers

RELAX
===

RELAX is a library that requires a small setup overhead in order to train and use a model. Focus on the implementation of your models and RELAX takes care of the rest. 

It follows the same syntax as the [haiku](https://github.com/deepmind/dm-haiku) and [optax](https://github.com/deepmind/optax), you just have to ```init``` the trainer params and then you can ```apply``` with your trained params the same way.

The [```Trainer```](https://github.com/romaingrx/relax/blob/35ecb13d34016f65de8de27b0695f111330118d6/relax/trainer.py#L63) class can train your model in a single line, just define your model, your loss and pass them to the [train](https://github.com/romaingrx/relax/blob/35ecb13d34016f65de8de27b0695f111330118d6/relax/trainer.py#L80) method. It is possible to optimize the different steps of the training process, just pass the flags ```jit_update_step``` and/or ```jit_epoch_loop```.

Example 
---

Here is an example on how easy it is to train a CNN classifier.

``` python
@dataclass
class CNN(hk.Module):
    num_classes : int
    conv_dim : int = 32

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = hk.Conv2D(self.conv_dim, kernel_shape=(3, 3), stride=2)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(2*self.conv_dim, kernel_shape=(3, 3), stride=2)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)

        logits = hk.Linear(self.num_classes)(x)       
        
        return logits

@hk.transform
def model(x):
    return CNN(10)(x)

def loss_fn(params, rng, data) -> jnp.ndarray:
    logits = model.apply(params, rng, data)
    return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1) # log softmax crossentropy
    
optimizer = optax.adam(0.001)

trainer = Trainer(model, optimizer, config)

init_rng = jax.random.PRNGKey(42)
fake_input = jnp.zeros(input_shape)
init_state = trainer.init(rng, fake_input)  

trained_state = trainer.train(init_state, loss_fn, ds, jit_update_step=True)
```
