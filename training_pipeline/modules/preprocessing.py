import tensorflow as tf

def get_prompt(x):
    system_prompt = """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.

"""    
    
    def get_prompt_with_input():        
        result = tf.strings.join(["### Instruction:\n", x['instruction']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Input:\n"])
        result = tf.strings.join([result, x['input']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Response:\n"])
        result = tf.strings.join([result, x['output']])
        result = tf.strings.join([result, "<|endoftext|>"])
        
        return tf.strings.join([system_prompt, result])

    def get_prompt_without_input():
        result = tf.strings.join(["### Instruction:\n", x['instruction']])
        result = tf.strings.join([result, '\n\n'])

        result = tf.strings.join([result, "### Response:\n"])
        result = tf.strings.join([result, x['output']])
        result = tf.strings.join([result, "<|endoftext|>"])
        
        return tf.strings.join([system_prompt, result])

    result = tf.cond(
        tf.math.equal(x['input'], ''),
        get_prompt_with_input,
        get_prompt_without_input
    )

    print(result)
    
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