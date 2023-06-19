import tensorflow as tf

def get_prompt(x):
    def get_prompt_with_input():
        result = tf.strings.join(["### Instruction:\n", x['instruction']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Input:\n"])
        result = tf.strings.join([result, x['input']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Response:\n"])
        result = tf.strings.join([result, x['output']])
        return result

    def get_prompt_without_input():
        result = tf.strings.join(["### Instruction:\n", x['instruction']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Response:\n"])
        result = tf.strings.join([result, x['output']])
        return result

    result = tf.cond(
        tf.math.equal(x['input'], ''),
        get_prompt_with_input,
        get_prompt_without_input
    )

    # ignore instruction and input returning values. they are just there 
    # since tf.map_fn requires the same structure as input in return
    return {
        'instruction': x['instruction'],
        'input': x['input'],
        'output': result
    }

def preprocessing_fn(inputs):
    inputs = tf.map_fn(lambda x: get_prompt(x), inputs)
    
    # final output will be mapped to `combine`
    return {
        'combine': inputs['output']
    }