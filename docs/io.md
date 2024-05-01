# Understanding Glinthawk input/output

Glinthawk uses [JSONL](https://jsonlines.org/) to store input and output data.
JSONL is a format that stores JSON objects in a file, one object per line.

As the user, you usually don't have to write your own input files; simply put
all the prompts in individual text files (`.txt`) in a directory, and use
`preprocess.py` to generate the input JSONL file.

## Input

Each prompt is stored as a JSON object with the following fields:

Field | Type | Values | Description
--- | --- | --- | ---
`temperature` | float | [0..255] | Temperature, will be divided by 255.f
`max_tokens` | int | [1..4096] | Maximum number of tokens to generate
`prompt_text` | string | | Prompt text
`user_data` | string | | Arbitrary string carried through to the output

Only `prompt_text` is required. The other fields are optional.

## Output

Each response includes all the input fields, plus the following:

Field | Type | Description
--- | --- | ---
`completion` | int[] | Token IDs of the generated text
`completion_text` | string | Generated text, detokenized by `postprocess.py`
