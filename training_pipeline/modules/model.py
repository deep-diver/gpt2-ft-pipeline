from tensorflow import keras
import keras_nlp

def get_gpt2_model(cardinality):
    gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_large_en")
    gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_large_en",
        sequence_length=256,
        add_end_token=True,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_large_en", preprocessor=gpt2_preprocessor
    )
    
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=cardinality,
        end_learning_rate=0.0,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    gpt2_lm.compile(
        optimizer=keras.optimizers.AdamW(learning_rate),
        loss=loss,
        weighted_metrics=["accuracy"],
        sampler="random"
    )    
    
    return gpt2_lm