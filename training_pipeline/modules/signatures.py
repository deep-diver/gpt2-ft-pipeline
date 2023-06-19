import tensorflow as tf

signature_dict = {
    "prompt": tf.TensorSpec(shape=[], dtype=tf.string, name="prompt"),
    "max_length": tf.TensorSpec(shape=[], dtype=tf.int64, name="max_length"),
}

def model_exporter(model):
  @tf.function(input_signature=[signature_dict])
  def serving_fn(inputs):
    prompt = tf.convert_to_tensor(inputs["prompt"])
    input_is_scalar = prompt.shape.rank == 0
    prompt = prompt[tf.newaxis] if input_is_scalar else prompt
    prompt = model.preprocessor.tokenizer(prompt)

    # Pad ragged to dense tensors.
    padded_shape = (1, inputs["max_length"])
    min_length = tf.reduce_min(prompt.row_lengths())
    input_mask = tf.ones_like(prompt, tf.bool).to_tensor(shape=padded_shape)
    prompt = prompt.to_tensor(shape=padded_shape)
    prompt = tf.cast(prompt, dtype="int64")

    generate_function = model.make_generate_function()
    output = generate_function({"token_ids": prompt, "padding_mask": input_mask}, min_length)

    token_ids, padding_mask = output["token_ids"], output["padding_mask"]
    padding_mask = padding_mask & (token_ids != model.preprocessor.tokenizer.end_token_id)
    token_ids = tf.ragged.boolean_mask(token_ids, padding_mask)

    token_ids = tf.cast(token_ids, dtype="int32")
    unicode_text = tf.strings.reduce_join(
        model.preprocessor.tokenizer.id_to_token_map.lookup(token_ids), axis=-1
    )
    split_unicode_text = tf.strings.unicode_split(unicode_text, "UTF-8")
    byte_text = tf.strings.reduce_join(
        model.preprocessor.tokenizer.unicode2byte.lookup(split_unicode_text), axis=-1
    )
    byte_text = tf.concat(byte_text, axis=0)
    byte_text = tf.squeeze(byte_text, 0)
    return {"result": byte_text}

  return serving_fn
